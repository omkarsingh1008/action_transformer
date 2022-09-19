
import numpy as np

head = h = [0, 15, 16, 1, 14]
right_foot = rf = [11, 12, 13,7]
left_foot = lf = [8, 9, 10,4]
prune= [1,8]

center_1 = 1
center_2 = 1
module_keypoint_1 = 8
module_keypoint_2 = 8


def add_velocities_single(seq,overwrite=False):
 
        
        seq_list = []
  
        
        v1 = np.zeros((30+1, seq.shape[1], 3-1))
        v2 = np.zeros((30+1, seq.shape[1], 3-1))
        
        v1[1:,...] = seq[:,:,:2]
        v2[:30,...] = seq[:,:,:2]
        vel = (v2-v1)[:-1,...]
        data = np.concatenate((seq[:,:,:2], vel), axis=-1)
        data = np.concatenate((data, seq[:,:,-1:]), axis=-1)       
        seq_list.append(data)

        return np.array(seq_list)

def scale_and_center(X_train):    
    for X in [X_train]:
        seq_list = []
        for seq in X:
            pose_list = []
            for pose in seq:
                zero_point = (pose[center_1, :2] + pose[center_2,:2]) / 2
                module_keypoint = (pose[module_keypoint_1, :2] + pose[module_keypoint_2,:2]) / 2
                scale_mag = np.linalg.norm(zero_point - module_keypoint)
                if scale_mag < 1:
                    scale_mag = 1
                pose[:,:2] = (pose[:,:2] - zero_point) / scale_mag
                pose_list.append(pose)
            seq = np.stack(pose_list)
            seq_list.append(seq)
        X = np.stack(seq_list)
    
    X_train = np.delete(X_train, prune, 2)
    scaled = True
    return X_train

def remove_confidence(X_train):
    X_train = X_train[...,:-1]
    return X_train

def flatten_features(X_train):
    X_train = X_train.reshape(X_train.shape[0], 30, -1)
    return X_train


def pre(x_train):
    x_train = add_velocities_single(x_train)
    x_train=np.array([x_train[0][:,0:15]])
    x_train = scale_and_center(x_train)
    x_train = remove_confidence(x_train)
    x_train = flatten_features(x_train)
    return x_train