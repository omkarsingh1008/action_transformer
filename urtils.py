import cv2
import numpy as np
from openvino.inference_engine import IECore
#from reid import ids_feature_,distance_,distance_list,ids_feature_list
import torch
import math
ie = IECore()
def load(filename,num_sources = 1):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)

def preprocess(frame,size):
    n,c,h,w = size
    try:
        input_image = cv2.resize(frame, (w,h))
    except:
        input_image = np.zeros((512,512,3))
        input_image = cv2.resize(input_image, (w,h))
    input_image = input_image.transpose((2,0,1))
    input_image.reshape((n,c,h,w))
    return input_image

def postprocess(frame,results):
    boxes = []
    labels = []
    scores = []
    for _, label, score, xmin, ymin, xmax, ymax in results:
        h1, w1 = frame.shape[:2]
        
        boxes.append(list(map(int, (xmin * w1, ymin * h1, (xmax - xmin)*w1, (ymax - ymin) * h1))))
        labels.append(int(label))
        scores.append(float(score))
    indices = cv2.dnn.NMSBoxes(bboxes=boxes, scores=scores, score_threshold=0.6, nms_threshold=0.7)
    

    boxes=[(boxes[idx]) for idx in indices]
    labels=[labels[idx] for idx in indices]
    scores=[scores[idx] for idx in indices]
    #for  box in boxes:
    
    #    cv2.rectangle(img=frame, pt1=box[:2], pt2=box[2:], color=(0,255,0), thickness=3)

    
    return boxes,scores,labels,frame
    
def xywh2xyxy(xywh):

    if len(xywh.shape) == 2:
        x = xywh[:, 0] + xywh[:, 2]
        y = xywh[:, 1] + xywh[:, 3]
        xyxy = np.concatenate((xywh[:, 0:2], x[:, None], y[:, None]), axis=1).astype('int')
        return xyxy
    if len(xywh.shape) == 1:
        x, y, w, h = xywh
        xr = x + w
        yb = y + h
        return np.array([x, y, xr, yb]).astype('int')

def draw_tracks(image, tracks,ids):
    """it wil take tracks and convert them into xmin,ymin, xmax,ymax

    Args:
        image ([array]): image
        tracks ([track]): tracks with xmin,ymin,width,height
        ids ([dic]): tracks with xmin,ymin, xmax,ymax

    Returns:
        image: image
        id : tracks
    """
    for trk in tracks:

        trk_id = trk[1]
        xmin = trk[2]
        ymin = trk[3]
        width = trk[4]
        height = trk[5]
        
        #xmax = xmin + width
        #ymax = ymin + height
        b=np.array([xmin,ymin, width, height])
        bbox=xywh2xyxy(b)

        xcentroid, ycentroid = int(xmin + 0.5*width), int(ymin + 0.5*height)

        text = "ID {}".format(trk_id)

        cv2.putText(image, text, (xcentroid - 10, ycentroid - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(image, (xcentroid, ycentroid), 4, (0, 255, 0), -1)
        cv2.rectangle(img=image, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
        ids[trk_id]=bbox


        

    return image,ids

def load_reid(filename,num_sources = 2):
    filename_bin = filename.split('.')[0]+".bin"
    net = ie.read_network(model = filename,weights = filename_bin)
    input_layer = next(iter(net.inputs))
    n,c,h,w = net.inputs[input_layer].shape
    exec_net = ie.load_network(network=net,device_name="CPU",num_requests = num_sources)
    output_layer = next(iter(net.outputs))

    return exec_net,input_layer,output_layer,(n,c,h,w)


exec_net,input_layer,output_layer,size = load_reid("/home/ubuntu/smart_store/models/reid/person-reidentification-retail-0277/FP32/person-reidentification-retail-0277.xml")

def draw_rec(frame,track_id):
    for i,bbox in track_id.items():
        cv2.putText(frame, str(i), bbox[:2], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.rectangle(img=frame, pt1=bbox[:2], pt2=bbox[2:], color=(0,255,0), thickness=3)
    return frame

def distance_list(feature,feat1):
    """compute dictance between two images if distance is closer to 1 that mean image is same 

    Args:
        feature ([type]): [description]
        feat1 ([type]): [description]

    Returns:
        [type]: [description]
    """
    x = torch.tensor(feature[0]).unsqueeze(0)
    dis=[]
    for i in feat1:
        y = torch.tensor(i).unsqueeze(0)
        d=torch.cosine_similarity(x, y)[0].numpy()
        dis.append(d)
    
    return sum(dis)/len(dis)

def ids_feature_list(ids,frame):
    """this function take tracks and then caculate feature list for all tracks which use for reid.

    Args:
        ids ([dic]): tracks
        frame ([array]): image

    Returns:
        ids_feat: new dic with value as feature and keys as id
    """
    ids_feat={}
    for i,bbox in ids.items():
        img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        img = preprocess(img,size)
        infer_res = exec_net.start_async(request_id=0,inputs={input_layer:img})
        status=infer_res.wait()
        results = exec_net.requests[0].outputs[output_layer][0]
        ids_feat[i]=[results]
    return ids_feat


def tracklets(max_key,feature,track_id,tracklets_id):
    """this fuction managing tracklets

    Args:
        max_key ([int]): 
        feature ([array]): [description]
        track_id ([dic]): {id:image}
        tracklets_id ([dic]): {id:int}

    Returns:
        [type]: [description]
    """
    if max_key not in tracklets_id:
        tracklets_id[max_key]=0
    tracklets_id[max_key] = tracklets_id[max_key]+1

    track_id[max_key].insert(tracklets_id[max_key],feature[0])

    if tracklets_id[max_key] > 50:
        tracklets_id[max_key] = 0

    return tracklets_id,track_id


                

def tracking(ids,frame,tracks_id,tracklets_id,tracks_draw):
    """this function reid person

    Args:
        ids ([dic]): new tracks
        frame ([array]): image
        tracks_id ([dic]): history of all tracks
        tracklets_id ([dic]): tracklets for all id
        tracks_draw ([dic]): 

    Returns:
        trcaks_draw: [description]
    """
    ids_feat = ids_feature_list(ids,frame)

    if len(tracks_id)==0:
        tracks_id = ids_feat
        tracks_draw = ids
        for i in ids:
            tracklets_id[i] = 0
    else:
        for i,feature in ids_feat.items():
            dis={}
                #print(type(feature))
            for id,feat1 in tracks_id.items():
                    #print(len(feat1))
                d = distance_list(feature,feat1)
                    #print(d)
                dis[id]=d
            max_key = max(dis, key=dis.get)
            #print("first_id:-",i)
            #print("track_id:-",id)
            print("reid:-",max_key)
            #print("reid_dis:-",dis[max_key])
           # print("reid_dis:-",dis)
                
            if dis[max_key] > .5:
                    #tracks_id[max_key] = feature
                tracklets_id,tracks_id = tracklets(max_key,feature,tracks_id,tracklets_id)
                    #if max_key not in tracklets_id:
                   #     tracklets_id[max_key]=0
                   # tracklets_id[max_key] = tracklets_id[max_key]+1
                tracks_draw[max_key] = ids[i]
            else:
                tracks_id[len(tracks_id)+1] = feature
                tracks_draw[len(tracks_id)+1] = ids[i]
    return tracks_draw,tracks_id,tracklets_id


def postprocess_yolov5(x_shape, y_shape,results):
    """this function postprocess detection and fillter only person

    Args:
        x_shape ([int]): image shape with x axis
        y_shape ([int]): image shape with y axis
        results ([array]): detection results

    Returns:
        boxes_person,scores_person,labels_person,boxes_bottel,scores_bottel,labels_bottel: fillter detection
    """
    boxes_person = []
    labels_person = []
    scores_person = []
    boxes_bottel = []
    labels_bottel = []
    scores_bottel = []
    labels,scores, cord = results.xyxyn[0][:, -1],results.xyxyn[0][:, -2], results.xyxyn[0][:, :-2]
    for i,l in enumerate(labels):
        if int(l) == 0:
            bbox = cord[i]
            boxes_person.append([int(bbox[0] * x_shape), int(bbox[1] * y_shape), int((bbox[2]-bbox[0]) * x_shape), int((bbox[3]-bbox[1]) * y_shape)])
            labels_person.append(int(labels[i]))
            scores_person.append(float(scores[i]))
        if int(l) == 39:
            bbox = cord[i]
            boxes_bottel.append([int(bbox[0] * x_shape), int(bbox[1] * y_shape), int((bbox[2]-bbox[0]) * x_shape), int((bbox[3]-bbox[1]) * y_shape)])
            labels_bottel.append(int(labels[i]))
            scores_bottel.append(float(scores[i]))
    return boxes_person,scores_person,labels_person,boxes_bottel,scores_bottel,labels_bottel

def pointInRect(point,rect):
    x1, y1, x2, y2 = rect
    x, y = point
    if (x1 < x and x < x2):
        if (y1 < y and y < y2):
            return 1
    return 0

def check_id(bottle,keypoint,tracks_draw):
      pick_id={}
      x_b,y_b = bottle[0][2:]
      keypoint_10 = keypoint[10]
      keypoint_9 = keypoint[9]
      dis_10 = np.sqrt((x_b-keypoint_10[0])**2+(y_b-keypoint_10[1])**2)
      dis_9 = np.sqrt((x_b-keypoint_9[0])**2+(y_b-keypoint_9[1])**2)
      print(dis_10)
      print(dis_9)
      if dis_10 <=250 or dis_9 <=250:
          for id,bbox in tracks_draw.items():
              flag = pointInRect(keypoint[1],bbox)
              pick_id[id] = flag
              print("*"*50)
              print(flag)
              print("*"*50)
      return pick_id
    

def check_id_m(bottle,keypoints,tracks_draw):
    pick_id={}
    for i,bol in enumerate(bottle):
        x_b,y_b = bol[2:]
        for keypoint in keypoints:
            keypoint_10 = keypoint[10]
            keypoint_9 = keypoint[9]
            dis_10 = np.sqrt((x_b-keypoint_10[0])**2+(y_b-keypoint_10[1])**2)
            dis_9 = np.sqrt((x_b-keypoint_9[0])**2+(y_b-keypoint_9[1])**2)
            for id,bbox in tracks_draw.items():
              flag = pointInRect(keypoint[1],bbox)
              pick_id[str(id)+str(i)] = flag
              print("*"*50)
              print(flag)
              print("*"*50)
    return pick_id


def id_assign(center_right,poses,output_transform,id,bottle_number):
    """this fuction calculate distance between bottle and wrist point if distance is less than 150 that mean perosn pick up a bottle

    Args:
        center_right ([list]): x,y
        poses ([array]): [description]
        output_transform ([]): [description]
        id ([type]): [description]
        bottle_number ([type]): [description]

    Returns:
        [type]: [description]
    """
    
    
    for i,pose in enumerate(poses):
        #print()
        dis=[]
        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        print("id:-",i)
        print("10:-",points[10])
        print("9:-",points[9])
        print(center_right)
        print(np.linalg.norm(center_right - points[10]))
        print(np.linalg.norm(center_right - points[9]))
        print("*"*50)
        dis_10 = np.linalg.norm(center_right - points[10])
        dis_9 = np.linalg.norm(center_right - points[9])
        dis.append(dis_10)
        dis.append(dis_9)
        min_key=dis.index(min(dis))
        if dis[min_key] < 150:
            id[str(i)+str(bottle_number)]="pick"+str(bottle_number)
    return id

def check_person(pickup_draw,id_bottle,poses,output_transform,tracks_draw):
    for i in id_bottle.keys():
        pose = poses[int(i[0])]
        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        point_0 = points[0]
   
        for id,bbox in tracks_draw.items():
            flag = pointInRect(point_0,bbox)
            if flag==1:
                pickup_draw[str(id)+i[-1]]=flag
    return pickup_draw

def many_bottel_pickup(pick_total,pickup_draw):
    for pick,flag in pickup_draw.items():
        if pick[:-1] not in pick_total:
            pick_total[pick[:-1]]=flag
        else:
            pick_total[pick[:-1]]=flag+ pick_total[pick[:-1]]
    return pick_total