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

"""Uses random data to test flow generated models for specific bands"""

def plot_model(i,n,conditional,band,bandindex,t_d,scaling_constant,curve):
    """Plots a model.
    - i = number in iteration
    - n = total number of iterations
    - conditional = m1,m2,l1,l2 values
    - band = the band of model
    - bandindex = the index in an array [g,r,i,z]
    - t_d = time (days) of lightcurves
    - scaling_constant = scaling constant of the curve
    - curve = lightcurve given by original data"""

    #set colour of model for ease of seeing
    c = cmap(i/n)

    #generate conditional array, I can't remember why it has to be so awkward but it works
    test_array = []
    for j in np.arange(n):
        temp = conditional[j]
        test_array.append(temp)
    test_array = np.array(test_array)
    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)

    #get samples from the model
    with torch.no_grad():
        samples = flow.sample(n,conditional = conditional)
    samples = samples.cpu().numpy()
    conditional = conditional.cpu().numpy()

    #get the conditions
    m1,m2,l1,l2 = conditional[i]
    lc = curve[i]*scaling_constant
    lc = np.nan_to_num(lc)

    #generate lightcurves using the original model
    lc_gen = Generate_LightCurve(m1,m2,l1,l2)[1]
    lc_gen = np.nan_to_num(lc_gen)

    #plot all three
    plt.plot(t_d,lc,label = f"Model[{i}]",c = c)
    plt.plot(lc_gen[0],lc_gen[1][bandindex+1],linestyle = "--",label = f"Model OG[{i}]",c = c)
    plt.plot(t_d,scaling_constant*samples[i],".",ms =5,label = f"Flow[{i}]",c = c)
    
def random_data(N,band,file,N_Samples = 100):
    """select random data
    - N = number of lines to draw at once
    - band = band being drawn
    - file = location where training data can be found
    - N_Samples = number of samples taken for a single curve in the flow machine"""

    #use this to find the scaling constant, might be a better way but this is mostly copy/pasted troubleshooting code
    fname = "Data_Cache/combined_nannum.pkl"
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

    #check the scaling constant worked
    try:
        assert np.max(curve) <= 1.0
    except:
        print("Curve not normalised correctly, curve max was:\t",np.max(curve))

    #get the conditions loaded from the data
    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    conditional = np.vstack((m1,m2,l1,l2)).T
    
    print(len(m1)," Data points")
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])

    #label axes
    plt.ylabel("Absolute Magnitude")
    plt.xlabel("Time [days]")

    #get N random choices from the conditionals array
    test_array = []
    indices = []
    for n in np.arange(N):
        i = random.randint(0,len(m1))
        indices.append(i)
        temp = random.choice(conditional)
        test_array.append(temp)
    test_array = np.array(test_array)
    cond = torch.from_numpy(test_array.astype(np.float32)).to(device)
    
    Big_Samples = [] #Where we put all of our samples so we can find the averages and means later.
    
    with torch.no_grad():
        for i in np.arange(N_Samples):
            samples  = flow.sample(N,conditional = cond)
            Big_Samples.append(samples)
    for i in np.arange(len(Big_Samples)):
        Big_Samples[i] = Big_Samples[i].cpu().numpy()
    Big_Samples = np.array(Big_Samples)

    axis = 0 #makes sure we find the mean over each of the N curves individually rather than the total mean or mean between all 3 etc.
    final_samples = np.mean(Big_Samples,axis = axis)
    print(final_samples.shape)
    std = np.std(Big_Samples,axis = axis)
    #add 3 standard deviations to get a region of +/- 3sigma
    max_lines = final_samples + 3*std 
    min_lines = final_samples - 3*std

    cond = cond.cpu().numpy()

    #colourmap to use in the graphs
    cmap = cm.get_cmap('viridis')

    for n in np.arange(N):
        #colour of graph
        col = cmap(n/N)

        #conditions of this graph
        m1_,m2_,l1_,l2_ = cond[n]

        #generate ligthcurve traditionally
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        lc = np.nan_to_num(lc)

        #plot the graphs
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
    random_data(3,"g","Data_Cache/combined_nannum.pkl",100)
