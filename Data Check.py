import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from matplotlib import cm

from glasflow import RealNVP
import torch
from torch import optim

import random
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DU17_Model import Generate_LightCurve

cmap =cm.get_cmap('hsv')
plt.style.use('seaborn-colorblind')
training_data = os.listdir("DU17_Training")
band = 'g'

fname = "Data_Cache/Crop/DU17_g_cropped.pkl" 
data = pd.read_pickle(fname)
#data = shuffle(data)

def check_data(band,data):
    bandindex = ['g','r','i','z'].index(band) + 1
    
    
    curve = data[band].values
    curve = np.vstack(curve)
    #curve = np.nan_to_num(curve)
    
    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    conditional = np.vstack((m1,m2,l1,l2)).T
    
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])

    ind = random.randint(0,len(curve))
    m1_i,m2_i,l1_i,l2_i = conditional[ind]
    curve_i = curve[ind]

    lc = Generate_LightCurve(m1_i,m2_i,l1_i,l2_i)[1]
    lc = np.nan_to_num(lc)

    #lc2 = Generate_LightCurve(m1_i,m2_i,l1_i,l2_i)[1]
    #lc2 = np.nan_to_num(lc2)

    #lc3 = Generate_LightCurve(m1_i,m2_i,l1_i,l2_i)[1]
    #lc3 = np.nan_to_num(lc3)

    #lc4 = Generate_LightCurve(m1_i,m2_i,l1_i,l2_i)[1]
    #lc4 = np.nan_to_num(lc4)

    plt.plot(lc[0],lc[1][bandindex],label = "Model",c = "red")
    plt.plot(t_d,curve_i,label = "training data")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

while True:
    check_data(band,data)
