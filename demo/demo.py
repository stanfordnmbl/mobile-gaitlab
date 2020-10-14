#!/usr/bin/env python
# coding: utf-8

# # Demonstration of Video Gait Analysis
# 
# In this notebook we present how to run OpenPose processing on a video and how apply neural networks from the paper to data processed by OpenPose. As a result, for a given mp4 file we will get predictions from all models.
# 
# To run this script you will need packages from the file `requirements.txt`. To install requirements run:
# ```bash
# pip install -r requirements.txt
# ```
# we recommend using conda or a virtual environment.
# 
# We start with some definitions and global constants.

# In[1]:


import pandas as pd
import numpy as np
import os
import json
from video_process_utils import *
from keras.models import load_model
import keras.losses
import keras.metrics
keras.losses.loss = keras.losses.mse
keras.metrics.loss = keras.metrics.mse
from statsmodels.regression.linear_model import OLSResults

from keras.backend.tensorflow_backend import clear_session
import gc
import tensorflow

# Reset Keras Session
def reset_keras():
    clear_session()

    try:
        del classifier # this is from global space - change this as you need
    except:
        pass

    print(gc.collect()) # if it's done something you should see a number being outputted
    
def convert_json2csv(json_dir):
    resL = np.zeros((300,75))
    resL[:] = np.nan
    for frame in range(1,300):
        test_image_json = '%sinput_%s_keypoints.json' %            (json_dir, str(frame).zfill(12))

        if not os.path.isfile(test_image_json):
            break
        with open(test_image_json) as data_file:  
            data = json.load(data_file)

        for person in data['people']:
            keypoints = person['pose_keypoints_2d']
            xcoords = [keypoints[i] for i in range(len(keypoints)) if i % 3 == 0]
            counter = 0
            resL[frame-1,:] = keypoints
            break

    #we can save space by dropping rows after the last row that isn't all nan
    check = np.apply_along_axis(lambda x: np.any(~np.isnan(x)),1,resL)
    for i in range(len(check)-1,-1,-1):
        if check[i]:
            break
    return resL[:i+1]

def get_prediction(centered_filtered, col, side = None):
    model = load_model("models/{}_best.pb".format(col))
    correction_model = OLSResults.load("models/{}_correction.pb".format(col))

    maps = {
        "KneeFlex_maxExtension": (-29.4408212510502, 114.8431545843835),
        "GDI": (36.314492983907, 77.03271217530302), # singlesided
        "gmfcs": (1, 3),
        "speed": (0.0718863507111867, 1.5259117583433834),
        "cadence": (0.222, 1.71556665023985),
        "SEMLS_dev_residual": (-0.8205001909638112, 3.309054961371647)
    }

    def undo_scaling(y,target_min,target_range):
        return y*target_range+target_min

    preds = []

    video_len = centered_filtered.shape[0]
    
    cols = x_columns
    if side == "L":
        cols = x_columns_left
    if side == "R":
        cols = x_columns_right

    samples = []
    for nstart in range(0,video_len-124,31):
        samples.append(centered_filtered[nstart:(nstart+124),cols])
        X = np.stack(samples)
        
    p = model.predict(X)[:,0]
    p = undo_scaling(p, maps[col][0], maps[col][1])
    p = np.transpose(np.vstack([p,np.ones(p.shape[0])]))
    p = correction_model.predict(pd.DataFrame(p))

#    reset_keras()# Shouldn't be needed anymore

    return np.mean(p)


# Next, we define a function which will run all models from the paper one by one:

# In[9]:


def get_all_preds(centered_filtered, centered_filtered_noswap):
    cols = ["GDI","gmfcs","speed","cadence","SEMLS_dev_residual"]
    return dict([(col, get_prediction(centered_filtered, col)) for col in cols] + [
        ("KneeFlex_maxExtension_L", get_prediction(centered_filtered_noswap, "KneeFlex_maxExtension", "L")),
        ("KneeFlex_maxExtension_R", get_prediction(centered_filtered_noswap, "KneeFlex_maxExtension", "R")),
    ])

# def predict(path):
#     values = {"test": 1}

#     return values, open(path, "rb")

def predict(path):
    os.system('cd /openpose ; /openpose/build/examples/openpose/openpose.bin --video {} --display 0 --write_json /gaitlab/output/keypoints -write_video /gaitlab/output/video.mp4 ; cd /gaitlab'.format(path))

    frames = convert_json2csv("output/keypoints/")

    centered_filtered = process_video_and_add_cols(frames)
    centered_filtered_noswap = process_video_and_add_cols(frames, swap_orientation=False)

    return get_all_preds(centered_filtered, centered_filtered_noswap), open("/gaitlab/output/video.mp4", "rb")

if __name__ == "__main__":
    # execute only if run as a script
    result, _ = predict("/gaitlab/input/input.mp4")

    print("Writing results to result.json...")
    text_file = open("/gaitlab/output/result.json", "w")
    text_file.write(json.dumps(result))
    text_file.close()
