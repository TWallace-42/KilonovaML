import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from matplotlib import cm

from glasflow import RealNVP
import torch
from torch import optim
import h5py 
import random
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#settings
plt.style.use('seaborn-colorblind')
cmap = cm.get_cmap('viridis')
N_samples = 100 #number of lines to generate

#parameters


"""Total_M = 2.83 #+0.04,-0.01 uncertainty

f_min = 0.40989399293
f_max = 0.56537102473 #mass is constrained between 1.16 and 1.6 => these are the constraints on f

f = random.uniform(f_min,f_max)#0.5 
m1 = f*Total_M
m2 = (1-f)*Total_M
#lambda (tidal deformabilities) is between 0 and 630
l1 = 300 #+420,-230 uncertainty
l2 = 300
"""

"""f = h5py.File("GW170817_GWTC-1.hdf5",'r')
#print(list(f.keys()))
data = f['IMRPhenomPv2NRT_highSpin_posterior']
m1_array = data['m1_detector_frame_Msun']
m2_array = data['m2_detector_frame_Msun']
l1_array = data['lambda1']
l2_array = data['lambda2']

m1 = np.mean(m1_array)
m2 = np.mean(m2_array)
l1 = np.mean(l1_array)
l2 = np.mean(l2_array)"""

with open('NS-priors.txt','r') as file:
    data2 = file.readlines()

final_dat2 = []
for d in data2:
    m1,m2,l1,l2 = d.split()[0:4]
    final_dat2.append([float(m1),
                       float(m2),
                       float(l1),
                       float(l2)])
    
final_dat2 = np.array(final_dat2)

m1_array = final_dat2[:,0]
m2_array = final_dat2[:,1]
l1_array = final_dat2[:,2]
l2_array = final_dat2[:,3]

m1 = np.mean(m1_array)
m2 = np.mean(m2_array)
l1 = np.mean(l1_array)
l2 = np.mean(l2_array)
print(f'm1: {m1}\tm2: {m2}\tl1: {l1}\tl2: {l2}')



conditionals = np.array([[m1,m2,l1,l2]])

cond = np.array([[m1,m2,l1,l2]])
#print(cond)
conditionals = np.vstack([m1_array,m2_array,l1_array,l2_array]).T

conditionals = np.reshape(conditionals,
                          (conditionals.shape[0],conditionals.shape[1],1))
#conditionals = np.array(conditionals)
#print(conditionals.shape)
#print(conditionals[0])
#input()
g_scaling = -14.019296288484181
r_scaling = -16.169625368881167
i_scaling = -17.85982432553296
z_scaling = -18.901707797571483
scales = [g_scaling,r_scaling,i_scaling,z_scaling]

labels = ["g","r","i","z"]
#Define model paths
g_fname = "Models/Model_G4/model_g.pth"
r_fname = "Models/Model_G4/model_r.pth"
i_fname = "Models/Model_G4/model_i.pth"
z_fname = "Models/Model_G4/model_z.pth"

#get time arrays
g_f = "Data_Cache/New/Comp_120_Original_nannum.pkl"
r_f = "Data_Cache/New/Comp_120_Original_nannum.pkl"
i_f = "Data_Cache/New/Comp_120_Original_nannum.pkl"
z_f = "Data_Cache/New/Comp_120_Original_nannum.pkl"

gdata = pd.read_pickle(g_f)
rdata = pd.read_pickle(r_f)
idata = pd.read_pickle(i_f)
zdata = pd.read_pickle(z_f)

t_g = np.vstack(gdata['time'])[0]
t_r = np.vstack(rdata['time'])[0]
t_i = np.vstack(idata['time'])[0]
t_z = np.vstack(zdata['time'])[0]
t_d = [t_g,t_r,t_i,t_z]
#Create flows
g_flow = torch.load(g_fname)
r_flow = torch.load(r_fname)
i_flow = torch.load(i_fname)
z_flow = torch.load(z_fname)

#send flows to device
device = torch.device('cuda')

g_flow.to(device)
r_flow.to(device)
i_flow.to(device)
z_flow.to(device)

g_flow.eval()
r_flow.eval()
i_flow.eval()
z_flow.eval()

flows = [g_flow,r_flow,i_flow,z_flow]

#Take Samples
G_Samples = []
R_Samples = []
I_Samples = []
Z_Samples = []
cond = torch.from_numpy(cond.astype(np.float32)).to(device)
conditionals = torch.from_numpy(conditionals.astype(np.float32)).to(device)
#print(conditionals)
place = 0
with torch.no_grad():
        #for i in np.arange(N_samples):
        
        for c in conditionals:
            #print(c)
            if c[1] >= 1:
                print("place:",place,"/",len(conditionals))
                place += 1
                c = c.T
                #print(cond)
                #print(conditionals)
                g  = g_flow.sample(1,conditional = c)
                r = r_flow.sample(1,conditional = c)
                i = i_flow.sample(1,conditional = c)
                z = z_flow.sample(1,conditional = c)
                G_Samples.append(g)
                R_Samples.append(r)
                I_Samples.append(i)
                Z_Samples.append(z)

for i in np.arange(len(G_Samples)):
    G_Samples[i] = G_Samples[i].cpu().numpy()
    R_Samples[i] = R_Samples[i].cpu().numpy()
    I_Samples[i] = I_Samples[i].cpu().numpy()
    Z_Samples[i] = Z_Samples[i].cpu().numpy()
Samples = [G_Samples,
           R_Samples,
           I_Samples,
           Z_Samples]

Ranges = []
Maxes = []
Mins = []


Samples2 = [[],
            [],
            [],
            []]
i = 0
"""for s in Samples:
    print(i)
    for l in s:
        M = np.max(l)
        m = np.min(l)
        ran = M-m
        if ran < 2e1:
            Ranges.append(ran)
            Maxes.append(M)
            Mins.append(m)
            Samples2[i].append(l)
    Samples2[i] = np.array(Samples2[i])
    i += 1"""

#Samples2 = np.array(Samples2)
#print(Samples2)      
plt.hist(Ranges,bins = 100)
plt.title("Ranges")
plt.yscale('log')
plt.show()
plt.hist(Maxes,bins = 100)
plt.title("Max Value")
plt.yscale('log')
plt.show()
plt.hist(Mins,bins = 100)
plt.title("Min Value")
plt.yscale('log')
plt.show()
#prepare for plotting
axis = 0
final_samples_g = np.mean(Samples2[0],axis = axis)[0]
final_samples_r = np.mean(Samples2[1],axis = axis)[0]
final_samples_i = np.mean(Samples2[2],axis = axis)[0]
final_samples_z = np.mean(Samples2[3],axis = axis)[0]


lines = [final_samples_g,
         final_samples_r,
         final_samples_i,
         final_samples_z]

stdg = 3*np.std(Samples2[0],axis = axis)[0]
stdr = 3*np.std(Samples2[1],axis = axis)[0]
stdi = 3*np.std(Samples2[2],axis = axis)[0]
stdz = 3*np.std(Samples2[3],axis = axis)[0]

max_g = final_samples_g + stdg
min_g = final_samples_g - stdg
max_r = final_samples_r + stdr
min_r = final_samples_r - stdr
max_i = final_samples_i + stdi
min_i = final_samples_i - stdi
max_z = final_samples_z + stdz
min_z = final_samples_z - stdz

max_lines = [max_g,max_r,max_i,max_z]
min_lines = [min_g,min_r,min_i,min_z]

for i in np.arange(len(max_lines)):
    #print(labels[i])
    col = cmap(i/len(max_lines))
    plt.plot(t_d[i],scales[i]*lines[i],"-",ms = 4, label = labels[i],c = col)
    plt.fill_between(t_d[i],min_lines[i]*scales[i],
                     max_lines[i]*scales[i],alpha = 0.2,
                     color = col)

plt.title(f'm1: {m1:.3g}, m2: {m2:.3g}, l1: {l1:.3g}, l2: {l1:.3g}')
plt.gca().invert_yaxis()
plt.legend()
plt.show()
