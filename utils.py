import numpy as np
import os
from collections import Counter
import cv2

def calc_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    radians = np.arctan2(bc[1], bc[0]) - np.arctan2(ba[1], ba[0])
    angle = np.abs(np.degrees(radians))
    if angle > 180:
        angle = 360 - angle
    return angle

def features_to_extract (k):
    #k: keypoints (17,2) landmarks/joints
    #returns normalised angles
    # YOLO keypoint indices (COCO) 
    L_SHOULDER, L_ELBOW, L_WRIST = 5, 7, 9 
    R_SHOULDER, R_ELBOW, R_WRIST = 6, 8, 10 
    L_HIP, R_HIP = 11, 12
    L_KNEE, R_KNEE = 13,14

    #Centering and scaling
    hip_center = (k[L_HIP] + k[R_HIP])/2
    torso_length = np.linalg.norm(k[L_SHOULDER]-hip_center)+1e-6
    k = (k-hip_center)/torso_length

    #Angles created
    left_elbow = calc_angle(k[L_SHOULDER],k[L_ELBOW],k[L_WRIST])
    right_elbow = calc_angle(k[R_SHOULDER],k[R_ELBOW],k[R_WRIST])

    left_armpit = calc_angle(k[L_ELBOW],k[L_SHOULDER],k[L_HIP])
    right_armpit = calc_angle(k[R_ELBOW],k[R_SHOULDER],k[R_HIP])

    left_hip = calc_angle(k[L_SHOULDER], k[L_HIP], k[L_KNEE])
    right_hip = calc_angle(k[R_SHOULDER], k[R_HIP], k[R_KNEE])

    # Torso orientation
    torso_vec = k[L_SHOULDER]
    torso_angle = np.arctan2(torso_vec[1], torso_vec[0])

    # Pull-up cue
    wrists_above_shoulders = (
        (k[L_WRIST][1] < k[L_SHOULDER][1]) + (k[R_WRIST][1] < k[R_SHOULDER][1])
    ) / 2
        
    # Sit-up cue
    torso_compression = np.linalg.norm(torso_vec)

    features = [
        left_elbow, right_elbow,
        left_armpit, right_armpit,
        left_hip, right_hip,
        torso_angle,
        wrists_above_shoulders,
        torso_compression
    ]

    return np.array(features)

def make_window(features, window_size=30, step=5):
    windows=[]

    for i in range(0,len(features)-window_size + 1, step):
        windows.append(features[i:i+window_size])

    return windows

def build_dataset(video_files, labels,location='extracted_features', window_size=30, steps=10):
    X = []
    y = []
    video_ids = []

    for vid_id, (name, lbl) in enumerate(zip(video_files, labels)):
        data = np.load(os.path.join(location, name))

        windows = make_window(data, window_size, steps)
        np_windows = np.array(windows).reshape(len(windows), -1)

        X.append(np_windows)
        y.extend([lbl] * len(np_windows))
        video_ids.extend([vid_id] * len(np_windows))

    X = np.vstack(X)
    y = np.array(y)
    video_ids = np.array(video_ids)

    return X, y, video_ids


def build_lstm_dataset(video_files, labels, window_size=30, step=5):
    X, y, vid_ids = [], [], []

    for vid_id, (name, lbl) in enumerate(zip(video_files, labels)):
        data = np.load(os.path.join("extracted_features", name))

        windows = make_window(data, window_size, step)

        for w in windows:
            X.append(w)      # shape (30, 9)
            y.append(lbl)
            vid_ids.append(vid_id)

    return np.array(X), np.array(y), np.array(vid_ids)

def smooth_prediction(new_pred): 
    frame_history.append(new_pred) 
    return Counter(frame_history).most_common(1)[0][0]

def rescale_frame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)
    
    return cv2.resize(frame, dimensions, interpolation=cv2.INTER_AREA)


def count_pushup(elbow_angle, state, counter):
    """
    Push-up counter state machine
    """
    if elbow_angle < 85 and state == "UP":
        state = "DOWN"
        
    elif elbow_angle > 160 and state == "DOWN":
        state = "UP"
        counter += 1
        
    return state, counter


def count_pullup(elbow_angle, state, counter):
    """
    Pull-up counter state machine
    """
    if elbow_angle > 160 and state == "UP":
        state = "DOWN"

    elif elbow_angle < 80 and state == "DOWN":
        state = "UP"
        counter += 1
    
    return state, counter

def count_situp(hip_angle, state, counter):
    """
    Sit-up counter state machine
    """
    if hip_angle > 160 and state == "UP":
        state = "DOWN"

    elif hip_angle < 60 and state == "DOWN":
        state = "UP"
        counter += 1
    
    return state, counter

def smooth_prediction(new_pred, history):
    history.append(new_pred)
    return Counter(history).most_common(1)[0][0]
    