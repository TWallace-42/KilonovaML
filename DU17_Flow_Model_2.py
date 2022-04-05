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


def Plot_random_samples():
    #settings
    plt.style.use('seaborn-colorblind')
    cmap = cm.get_cmap('viridis')
    N_samples = 100 #number of lines to generate
    device = torch.device('cuda')


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
    g_flow = torch.load(g_fname)
    r_flow = torch.load(r_fname)
    i_flow = torch.load(i_fname)
    z_flow = torch.load(z_fname)

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

    m1 = np.array(gdata['m1'])
    m2 = np.array(gdata['m2'])
    l1 = np.array(gdata['l1'])
    l2 = np.array(gdata['l2'])
    index = random.randint(0,len(m1))
    m1 = m1[index]
    m2 = m2[index]
    l1 = l1[index]
    l2 = l2[index]

    cond = np.array([[m1,m2,l1,l2]])

    plt.title(f'm1: {m1:.3g}, m2: {m2:.3g}, l1: {l1:.3g}, l2: {l2:.3g}')
    print(cond)
    cond = torch.from_numpy(cond.astype(np.float32)).to(device)
    #send flows to device


    g_flow.to(device)
    r_flow.to(device)
    i_flow.to(device)
    z_flow.to(device)

    g_flow.eval()
    r_flow.eval()
    i_flow.eval()
    z_flow.eval()

    flows = [g_flow,r_flow,i_flow,z_flow]
    G_samples = []
    R_samples = []
    I_samples = []
    Z_samples = []

    N_samples = 100
    with torch.no_grad():
        for j in np.arange(N_samples):
            g = g_flow.sample(1,conditional = cond)
            r = r_flow.sample(1,conditional = cond)
            i = i_flow.sample(1,conditional = cond)
            z = z_flow.sample(1,conditional = cond)
            G_samples.append(g)
            R_samples.append(r)
            I_samples.append(i)
            Z_samples.append(z)
    for j in np.arange(len(G_samples)):
        G_samples[j] = G_samples[j].cpu().numpy()
        R_samples[j] = R_samples[j].cpu().numpy()
        I_samples[j] = I_samples[j].cpu().numpy()
        Z_samples[j] = Z_samples[j].cpu().numpy()
    g = np.mean(G_samples,axis = 0)
    r = np.mean(R_samples,axis = 0)
    i = np.mean(I_samples,axis = 0)
    z = np.mean(Z_samples,axis = 0)
    std_g = np.std(G_samples,axis = 0)
    std_r = np.std(R_samples,axis = 0)
    std_i = np.std(I_samples,axis = 0)
    std_z = np.std(Z_samples,axis = 0)

    max_g = g[0] + 3*std_g
    min_g = g[0] - 3*std_g
    max_r = r[0] + 3*std_r
    min_r = r[0] - 3*std_r
    max_i = i[0] + 3*std_i
    min_i = i[0] - 3*std_i
    max_z = z[0] + 3*std_z
    min_z = z[0] - 3*std_z

    max_lines = [max_g,max_r,max_i,max_z]
    min_lines = [min_g,min_r,min_i,min_z]
    final_samples = [g,r,i,z]
    cmap =cm.get_cmap('viridis')
    for n in np.arange(4):
                col = cmap(n/4)
                plt.plot(t_d[n],scales[n]*final_samples[n][0],"-",label = labels[n],
                         ms =4,c = col)
                plt.fill_between(t_d[n],min_lines[n][0]*scales[n],
                                 max_lines[n][0]*scales[n],alpha = 0.2,color = col)

    plt.xlabel('time [days]')
    plt.ylabel('Absolute Magnitude')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()
while True:
    Plot_random_samples()
