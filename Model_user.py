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



def plot_model(i,n,conditional,band,bandindex,t_d,scaling_constant,curve):
    c = cmap(i/n)
    test_array = []
    for j in np.arange(n):
        temp = conditional[j]
        test_array.append(temp)
    test_array = np.array(test_array)
    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)
    #print(conditional)
    with torch.no_grad():
        samples = flow.sample(n,conditional = conditional)
    samples = samples.cpu().numpy()
    conditional = conditional.cpu().numpy()

    m1,m2,l1,l2 = conditional[i]
    lc = curve[i]*scaling_constant
    
    lc = np.nan_to_num(lc)

    
    m1,m2,l1,l2 = conditional[i]
    #print(m1,m2,l1,l2)
    lc_gen = Generate_LightCurve(m1,m2,l1,l2)[1]
    lc_gen = np.nan_to_num(lc_gen)
    
    plt.plot(t_d,lc,label = f"Model[{i}]",c = c)
    plt.plot(lc_gen[0],lc_gen[1][bandindex+1],linestyle = "--",label = f"Model OG[{i}]",c = c)
    plt.plot(t_d,scaling_constant*samples[i],".",ms =5,label = f"Flow[{i}]",c = c)
    
def random_data(N,band,file,N_Samples = 100):

    fname = "Data_Cache/New/combined.pkl"
    data = pd.read_pickle(fname)
    data = shuffle(data)
    
    curve = data[band].values
    curve = np.vstack(curve)
    curve = np.nan_to_num(curve)

    scaling_constant = np.min(curve)
    print(f'scaling_constant {band}: {scaling_constant}')
    
    #Load Data
    bandindex = ['g','r','i','z'].index(band) + 1
    
    fname = file
    print(f'file: {fname}')

    data = pd.read_pickle(fname)
    data = shuffle(data)
        
    curve = data[band].values
    curve = np.vstack(curve)
    curve = np.nan_to_num(curve)
        
    curve = curve/scaling_constant
    try:
        assert np.max(curve) <= 1.0
    except:
        print("Curve not normalised correctly, curve max was:\t",np.max(curve))
    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    conditional = np.vstack((m1,m2,l1,l2)).T
    print(random.choice(conditional))
    print(len(m1)," Data points")
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])
        
    #plt.title(f'{band} band Flow vs Model')
    plt.ylabel("Absolute Magnitude")
    plt.xlabel("Time [days]")

    test_array = []
    indices = []
    for n in np.arange(N):
        i = random.randint(0,len(m1))
        indices.append(i)
        temp = random.choice(conditional)
        test_array.append(temp)
    test_array = np.array(test_array)
    cond = torch.from_numpy(test_array.astype(np.float32)).to(device)
    
    Big_Samples = []
    
    with torch.no_grad():
        for i in np.arange(N_Samples):
            samples  = flow.sample(N,conditional = cond)
            Big_Samples.append(samples)
    for i in np.arange(len(Big_Samples)):
        Big_Samples[i] = Big_Samples[i].cpu().numpy()
    Big_Samples = np.array(Big_Samples)

    axis = 0
    final_samples = np.mean(Big_Samples,axis = axis)
    print(final_samples.shape)
    std = np.std(Big_Samples,axis = axis)
    max_lines = final_samples + 3*std #np.max(Big_Samples,axis = 0)
    min_lines = final_samples - 3*std #np.min(Big_Samples,axis = 0)

    cond = cond.cpu().numpy()
    print(cond)
    cmap = cm.get_cmap('viridis')

    #t_d_scaled = np.linspace(t_d[0],t_d[-1],num = len(final_samples))
    for n in np.arange(N):
        col = cmap(n/N)
        m1_,m2_,l1_,l2_ = cond[n]
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        lc = np.nan_to_num(lc)
        plt.plot(lc[0],lc[1][bandindex],"--",label = f"[{m1_:.3g}, {m2_:.3g}, {l1_:.3g}, {l2_:.3g}]",c = col)
        plt.plot(t_d,scaling_constant*final_samples[n],"-",ms =4,c = col)
        plt.fill_between(t_d,min_lines[n]*scaling_constant,max_lines[n]*scaling_constant,alpha = 0.2,color = col)

    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

#Define the Model
#model_path = "Models/Model_G4/model_r.pth"
model_path = "Models/Model_G4/model_g.pth"
flow = torch.load(model_path)
device = torch.device('cuda')
flow.to(device)
flow.eval()

#colourmaps definitions
cmap =cm.get_cmap('hsv')
plt.style.use('seaborn-colorblind')

while True:
    #"DU17_training/Comp_91_Original_Combined_zero_comp.pkl",100)#
    random_data(3,"g","Data_Cache/New/DU17_15_nannum.pkl",100)
