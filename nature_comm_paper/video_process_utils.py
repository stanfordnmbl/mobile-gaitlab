
#utility functions for processing videos and extracting features
import numpy as np
import random as rn
import os
import matplotlib.pyplot as plt
import scipy
import math
import sys
import pandas as pd
from scipy.ndimage.filters import gaussian_filter1d

from sklearn.metrics import mean_squared_error
from scipy.stats import linregress

from scipy import interpolate
from scipy import signal
import pickle
import itertools


FPS = 25.0

NOSE = 0
NECK = 1
RSHO = 2
RELB = 3
RWRI = 4
LSHO = 5
LELB = 6
LWRI = 7
MHIP = 8
RHIP = 9
RKNE = 10
RANK = 11
LHIP = 12
LKNE = 13
LANK = 14
REYE = 15
LEYE = 16
REAR = 17
LEAR = 18
LBTO = 19
LSTO = 20
LHEL = 21
RBTO = 22
RSTO = 23
RHEL = 24

topoint = lambda x: range(2*x,2*x+2)

def flatten(lst):
    return list(itertools.chain.from_iterable(lst))

def count_nonnan(A):
    return np.count_nonzero(~np.isnan(A))

def num_nan_or_zero(A):
    return np.sum(np.isnan(A)) + np.sum(A == 0)

def expand_columns(col_lst):
    return flatten([[2*x,2*x+1] for x in col_lst])

def max_pct_nan_or_zero_given_cols(A,col_lst):
    col_indices_used = expand_columns(col_lst)
    A = A.copy()[:,col_indices_used]
    mask = (A==0) | (np.isnan(A))
    return np.max(np.sum(mask,axis=0))*1.0/len(A)

def first_row_not_all_nan_or_zero(A,col_lst):
    col_indices_used = expand_columns(col_lst)
    A = A.copy()[:,col_indices_used]
    mask = (A != 0) & (~np.isnan(A))
    mask = np.sum(mask,axis=1)
    return (mask != 0).argmax()

    
def drop_confidence_cols(res):
    res = res.copy()
    num_parts = res.shape[1]/3
    processed_cols = [True,True,False] * int(num_parts)
    return res[:,processed_cols]

def fill_nan(A):
    '''
    interpolate to fill nan values
    '''
    inds = np.arange(A.shape[0])
    good = np.where(np.isfinite(A))
    if(len(good[0]) <= 1):
        return A
    #linearly interpolate and then fill the extremes with the mean (relatively similar to)
    #what kalman does 
    f = interpolate.interp1d(inds[good], A[good],kind="linear",bounds_error=False)
    B = np.where(np.isfinite(A),A,f(inds))
    B = np.where(np.isfinite(B),B,np.nanmean(B))
    return B


def impute_frames(frames):
    return np.apply_along_axis(fill_nan,arr=frames,axis=0)


def filter_frames(frames):
    return np.apply_along_axis(lambda x: gaussian_filter1d(x,1),arr=frames,axis=0)

def preprocess_frames(res):
    res = res.copy()
    num_parts = res.shape[1]/3
    res = drop_confidence_cols(res)
    res[res == 0] = np.NaN
    
    res = impute_frames(res)
    res = filter_frames(res)
    
    mhip_x = ((res[:,2*RHIP] + res[:,2*LHIP])/2).reshape(-1,1)
    mhip_y = ((res[:,2*RHIP+1] + res[:,2*LHIP+1])/2).reshape(-1,1)
    mhip_coords = np.hstack([mhip_x,mhip_y]*int(num_parts))
    
    scale_vector_R = np.apply_along_axis(lambda x: np.linalg.norm(x[topoint(RHIP)] -
                                                                  x[topoint(RKNE)]),1,res)
    scale_vector_L = np.apply_along_axis(lambda x: np.linalg.norm(x[topoint(LHIP)] -
                                                                  x[topoint(LKNE)]),1,res)
    scale_vector = ((scale_vector_R + scale_vector_L)/2.0).reshape(-1,1)
    
    res = (res-mhip_coords)/scale_vector
    #apply the sign
    lt_x = res[:,2*LANK] - res[:,2*LBTO]
    rt_x = res[:,2*RANK] - res[:,2*RBTO]
    orientation = np.where(lt_x+rt_x >= 0,1,-1).reshape(-1,1)
    return res/orientation
    

def get_distance(A,B,centered_filtered):
    p_A = np.array([centered_filtered[:,2*A],centered_filtered[:,2*A+1]]).T
    p_B = np.array([centered_filtered[:,2*B],centered_filtered[:,2*B+1]]).T    
    p_BA = p_A - p_B
    return np.linalg.norm(p_BA,axis=1)
    
def get_angle(A,B,C,centered_filtered):
    """
    finds the angle ABC, assumes that confidence columns have been removed
    A,B and C are integers corresponding to different keypoints
    """
    p_A = np.array([centered_filtered[:,2*A],centered_filtered[:,2*A+1]]).T
    p_B = np.array([centered_filtered[:,2*B],centered_filtered[:,2*B+1]]).T
    p_C = np.array([centered_filtered[:,2*C],centered_filtered[:,2*C+1]]).T
    p_BA = p_A - p_B
    p_BC = p_C - p_B
    dot_products = np.sum(p_BA*p_BC,axis=1)
    norm_products = np.linalg.norm(p_BA,axis=1)*np.linalg.norm(p_BC,axis=1)
    return np.arccos(dot_products/norm_products)