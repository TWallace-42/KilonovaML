import numpy as np

import matplotlib
#matplotlib.use("Qt5Agg")

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
from scipy.optimize import curve_fit

from DU17_Model import Generate_LightCurve

#   Compares the time taken to generate lightcurves traditionally vs with flows

def time_model(N,band,file,N_Samples,m1,m2,l1,l2,
               conditional):
    "times the time taken to generate N lightcurves"

    #       Machine Learning Test   #
    #-------------------------------#
    
    #prepare the conditionals used
    test_array = []
    indices = []

    for n in np.arange(N):
        i = random.randint(0,len(m1))
        indices.append(i)
        temp = random.choice(conditional)
        test_array.append(temp)
    test_array = np.array(test_array)
    Big_Samples = []

    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)#send conditions to the device
    ML_start = time.time() # start timing here
    
    with torch.no_grad():
        for i in np.arange(N_Samples):
            samples  = flow.sample(N,conditional = conditional) #get samples for the curve
            Big_Samples.append(samples)
    for i in np.arange(len(Big_Samples)):
        Big_Samples[i] = Big_Samples[i].cpu().numpy()#send samples to cpu so we can work with them again
    Big_Samples = np.array(Big_Samples)

    final_samples = np.mean(Big_Samples,axis = 0) #get the mean of the many samples of curves
    std = np.std(Big_Samples,axis = 0)

    #get plus and minus one standard deviation
    max_lines = final_samples + std 
    min_lines = final_samples - std 
    ML_end = time.time() #stop timing ML
    
    ML_time = ML_end - ML_start #time for machine learning

    
    #   Traditional Model Test  #
    #---------------------------#
    conditional = conditional.cpu().numpy()
    Model_start = time.time() #start timing
    for n in np.arange(N):
        print(f'\r{100*n/N:.4f}%',end = "\r") #best run from cmd so this doesnt clog up the IDE screen, \r puts it back to start of line.
        col = cmap(n/N)
        m1_,m2_,l1_,l2_ = conditional[n]
        lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
        lc = np.nan_to_num(lc)
    Model_end = time.time() #stop testing

    #print result
    print()
    Model_time = Model_end - Model_start
    print(f'N_Samples = {N_Samples}')
    print(f'Machine took: {ML_time:.5f}ms')
    print(f'Model took: {Model_time:.5f}ms')
    print(f'Difference: {Model_time - ML_time} (Model - ML)')
    return(Model_time,ML_time)


#   Generate Timed Data #
#-----------------------#
# this section can be   #
# commented out once Mt #
# and MLt.npy have been #
# created.              #

#Define the Model
model_path = "Models/Model_G4/model_z.pth"
flow = torch.load(model_path,map_location = torch.device('cuda'))
device = torch.device('cuda')
flow.to(device)
flow.eval()

#colourmaps definitions
cmap =cm.get_cmap('hsv')
plt.style.use('seaborn-colorblind')

band = "z"
file = "Data_Cache/combined_nannum.pkl"
N = 200

#find the scaling constant 
fname = "Data_Cache/combined_nannum.pkl" #it is better to load this as your original data without any altering. hence why fname =/= file generally
data = pd.read_pickle(fname)
data = shuffle(data)
    
curve = data[band].values
curve = np.vstack(curve)
curve = np.nan_to_num(curve)

scaling_constant = np.min(curve)
    
#Load Data
bandindex = ['g','r','i','z'].index(band) + 1
    
fname = file #in general fname =/= file just when tidying this up it was easier.


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

m1 = np.vstack(data['m1'])
m2 = np.vstack(data['m2'])
l1 = np.vstack(data['l1'])
l2 = np.vstack(data['l2'])
    
    


#Arrays for time, Mt = Model Time, MLt = Machine Learning Time
Mt = []
MLt = []
for n in np.arange(1,N,10):
    
    print(f'========{n}=========')
    mt, mlt = time_model(int(n),band,
                         file,
                         100,m1,m2,l1,l2,conditional)
    Mt.append(mt)
    MLt.append(mlt)
Mt = np.array(Mt)
MLt = np.array(MLt)
np.save("Mt.npy",Mt)
np.save("MLt.npy",MLt)


#   Analysing Timings   #
#-----------------------#
#Just load the numpy    #
#files when it is done  #

"""#Once Mt,MLt are created once they can be loaded in after this rather than generating every time
def straight_line(x,m,c):
    return(m*x+c)


Mt = np.load("Mt.npy")
MLt = np.load("MLt.npy")

popt,pcov = curve_fit(straight_line,np.arange(1,1000,10),Mt)
popt2,pcov2 = curve_fit(straight_line,np.arange(1,1000,10),MLt)
print(popt,pcov)
print(popt2,pcov2)

plt.plot(np.arange(1,1000,10),Mt,".",label = f"DU17 Model, m = {popt[0]:.2g},c = {popt[1]:.2g}")
plt.plot(np.arange(1,1000,10),MLt,".",label = f"Machine Learning Model, m = {popt2[0]:.2g},c = {popt2[1]:.2g}")
plt.xlabel("Number of curves generated")
plt.ylabel("Time (s)")
plt.legend()
plt.show()"""
