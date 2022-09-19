import cv2

import numpy as np
from openvino.inference_engine import IECore
import openvino_models as models
from motrackers import CentroidTracker
import json
import requests

from preprocess__ import pre
from utils.tools import read_yaml
# GENERAL LIBRARIES 
import math
import numpy as np
import joblib
from pathlib import Path
# MACHINE LEARNING LIBRARIES
import sklearn
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_addons as tfa
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
# OPTUNA
import optuna
from optuna.trial import TrialState
from utils.transformer import TransformerEncoder, PatchClassEmbedding, Patches
from utils.data import load_mpose, random_flip, random_noise, one_hot
from utils.tools import CustomSchedule, CosineSchedule
from utils.tools import Logger

default_skeleton = ((15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6))

colors = (
        (255, 0, 0), (255, 0, 255), (170, 0, 255), (255, 0, 85),
        (255, 0, 170), (85, 255, 0), (255, 170, 0), (0, 255, 0),
        (255, 255, 0), (0, 255, 85), (170, 255, 0), (0, 85, 255),
        (0, 255, 170), (0, 0, 255), (0, 255, 255), (85, 0, 255),
        (0, 170, 255))

def draw_pose(img,poses, point_score_threshold,output_transform,skeleton=default_skeleton,draw_ellipses=False):
    """draw_pose is fuction which use for drawing pose on image

    Args:
        img ([array]): [description]
        poses ([array]): [description]
        point_score_threshold ([array]): [description]
        output_transform ([function]): [description]
        skeleton ([type], optional): [description]. Defaults to default_skeleton.
        draw_ellipses (bool, optional): [description]. Defaults to False.

    Returns:
        img: with draw poses
    """
    
    img = output_transform.resize(img)
    if poses.size == 0:
        return img
    stick_width = 4

    img_limbs = np.copy(img)
    for pose in poses:
        points = pose[:,:2].astype(np.int32)
        points = output_transform.scale(points)
        points_scores = pose[:,2]

        for i,(p,v) in enumerate(zip(points, points_scores)):
            if v > point_score_threshold:
                pass
        cv2.circle(frame,points[4],10,(0,0,0),-1)    
        for i, j in skeleton:
            if points_scores[i] > point_score_threshold and points_scores[j] > point_score_threshold:
                if draw_ellipses:
                    middle = (points[i] + points[j]) // 2
                    vec = points[i] - points[j]
                    length = np.sqrt((vec * vec).sum())
                    angle = int(np.arctan2(vec[1], vec[0]) * 180 / np.pi)
                    polygon = cv2.ellipse2Poly(tuple(middle), (int(length / 2), min(int(length / 50), stick_width)),
                                               angle, 0, 360, 1)
                    cv2.fillConvexPoly(img_limbs, polygon, colors[j])
                else:
                    cv2.line(img_limbs, tuple(points[i]), tuple(points[j]), color=colors[j], thickness=stick_width)
    cv2.addWeighted(img, 0.4, img_limbs, 0.6, 0, dst=img)
    return img


plugin_config = {'CPU_BIND_THREAD': 'NO', 'CPU_THROUGHPUT_STREAMS': 'CPU_THROUGHPUT_AUTO'}
ie= IECore()
model = models.OpenPose(ie, "human-pose-estimation-0001/FP32/human-pose-estimation-0001.xml", target_size=None, aspect_ratio=1,
                                prob_threshold=0.1)
input=model.image_blob_name
out_pool=model.pooled_heatmaps_blob_name
out_ht=model.heatmaps_blob_name
out_paf=model.pafs_blob_name
n,c,h,w = model.net.inputs[input].shape
exec_net = ie.load_network(network=model.net,config=plugin_config,device_name="CPU",num_requests = 1)


config = read_yaml('utils/config.yaml')

print(config)
model_size = config['MODEL_SIZE']
n_heads = config[model_size]['N_HEADS']
n_layers = config[model_size]['N_LAYERS']
embed_dim = config[model_size]['EMBED_DIM']
dropout = config[model_size]['DROPOUT']
mlp_head_size = config[model_size]['MLP']
activation = tf.nn.gelu
d_model = 64 * n_heads
d_ff = d_model * 4
labels = config["LABELS"]
 
split = 1
fold = 0
trial = None
bin_path = config['MODEL_DIR']

def build_act(transformer):
        inputs = tf.keras.layers.Input(shape=(config['FRAMES'], 
                                              config[config['DATASET']]['KEYPOINTS']*config['CHANNELS']))
        x = tf.keras.layers.Dense(d_model)(inputs)
        x = PatchClassEmbedding(d_model, config['FRAMES'])(x)
        x = transformer(x)
        x = tf.keras.layers.Lambda(lambda x: x[:,0,:])(x)
        x = tf.keras.layers.Dense(mlp_head_size)(x)
        outputs = tf.keras.layers.Dense(config['CLASSES'])(x)
        return tf.keras.models.Model(inputs, outputs)


transformer = TransformerEncoder(d_model, n_heads,d_ff, dropout, activation, n_layers)
model_act = build_act(transformer)

print(model_act.summary())

#lr = CustomSchedule(d_model, 
#             warmup_steps=len(ds_train)*config['N_EPOCHS']*config['WARMUP_PERC'],
#             decay_step=len(ds_train)*config['N_EPOCHS']*config['STEP_PERC'])

optimizer = tfa.optimizers.AdamW(weight_decay=config['WEIGHT_DECAY'])

model_act.compile(optimizer=optimizer,
                           loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1),
                           metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")])


model_act.load_weights("bin/AcT_micro_2_9.h5")


cap = cv2.VideoCapture("VID_20220719_175104.mp4")
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width,frame_height)
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
output = cv2.VideoWriter('output9.mp4', fourcc, fps, (1000,1000))
font = cv2.FONT_HERSHEY_SIMPLEX
l=[]
s=""
while True:
    _,frame = cap.read()
    frame = cv2.resize(frame, (1000,1000))
    x_shape, y_shape = frame.shape[1], frame.shape[0]
    output_transform = models.OutputTransform(frame.shape[:2], None)
    output_resolution = (frame.shape[1], frame.shape[0])
    inputs, preprocessing_meta = model.preprocess(frame)
    infer_res = exec_net.start_async(request_id=0,inputs={input:inputs["data"]})
    status=infer_res.wait()
    results_pool = exec_net.requests[0].outputs[out_pool]
    results_ht = exec_net.requests[0].outputs[out_ht]
    results_paf = exec_net.requests[0].outputs[out_paf]
    results={"heatmaps":results_ht,"pafs":results_paf,"pooled_heatmaps":results_pool}
    poses,scores=model.postprocess(results,preprocessing_meta)
    #print("*"*50)
    #print(poses)
    #s=""
    try:
        l.append(poses[0])
    except Exception as e:
        print(e)
    if len(l)==30:
        l = np.array(l)
        l = pre(l)
        print(l.shape)
        p = np.argmax(model_act.predict(l)) 
        s=labels[p]
        print(s)
        l=[]        
    #print("*"*50)
    #print(scores)
    #print("*"*50)
    frame = draw_pose(frame,poses,0.1,output_transform)
    if len(l) < 30:
        cv2.putText(frame, str(s), (100,100), 1, cv2.FONT_HERSHEY_DUPLEX, (0, 0, 255), 3)
    output.write(frame)
    cv2.imshow('smart store', frame)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
cap.release()
output.release()
cv2.destroyAllWindows()
