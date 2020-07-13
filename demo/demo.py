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

os.system('cd /openpose ; /openpose/build/examples/openpose/openpose.bin --video /gaitlab/input/input.mp4 --display 0 --write_json /gaitlab/output/keypoints -write_video /gaitlab/output/video.mp4 ; cd /gaitlab')

# ## Collect data
# 
# Record a video of gait from the side. To achieve best results:
# * Camera should be placed 4 meters from the line of walking
# * Ask the person to walk straight from the right to the left (looking from the camera perspective)
# * Make sure there are no other people in the background
# * If the line of walking is short, ask them to turn at the end and go back
# * Record in potrait mode
# * Follow the person by rotating the camera set in the same place
# * Save the video as mp4 at 30 frames per second
# 
# See an example video below.
# 
# ## Run openpose
# 
# In order to run openpose on your video, save your video as `input.mp4` in the `in` directory of this repository. You only need NVIDIA docker installed in your system -- OpenPose will be downloaded automatically.
# 
# For convenience, we added an example video to `in` directory.

# In[2]:


# ## Processing the output
# 
# Next, we need to process OpenPose output to the format accepted by our neural networks. This will include:
# 
# * processing json files to create a data matrix,
# * intrapolating signals,
# * scaling (normalizing) observations.
# 
# First, we convert JSON output to a data matrix

# In[4]:


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


# In[5]:


frames = convert_json2csv("output/keypoints/")
pd.DataFrame(frames)
print(frames)

# Next, we normalize frames using a predefined preprocessing function `process_video_and_add_cols` (see `video_process_utils.py` for details).

# In[6]:


processed_videos = []
processed_video_segments = []

centered_filtered = process_video_and_add_cols(frames)
centered_filtered_noswap = process_video_and_add_cols(frames,
                                swap_orientation=False) # don't swap X for metrics computed on both legs


# For validation, we plot some of the features. If everything is correct all signals should be smooth. We chose a dark theme 

# In[7]:


# The big discontinouity around the friame 170 corresponds to the change of the orientation of the video. We mirror videos so that they all seem to have a person walking from right to left.

# ## Predicting gait parameters
# 
# With preprocessed data we are finally ready to run predition models. First, we define a function that takes a preprocessed multivariate time series data and runs a selected model.

# In[8]:


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

    return np.mean(p)


# Next, we define a function which will run all models from the paper one by one:

# In[9]:


def get_all_preds(centered_filtered):
    cols = ["GDI","gmfcs","speed","cadence","SEMLS_dev_residual"]
    return dict([(col, get_prediction(centered_filtered, col)) for col in cols] + [
        ("KneeFlex_maxExtension_L", get_prediction(centered_filtered_noswap, "KneeFlex_maxExtension", "L")),
        ("KneeFlex_maxExtension_R", get_prediction(centered_filtered_noswap, "KneeFlex_maxExtension", "R")),
    ])
    


# Finally, we run all models on our data:

# In[10]:


print(get_all_preds(centered_filtered))

