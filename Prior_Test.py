import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#plt.style.use('seaborn-colorblind')

import h5py
import random
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle




f = h5py.File("GW170817_GWTC-1.hdf5",'r')


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

data = f['IMRPhenomPv2NRT_highSpin_posterior']
m1_array = data['m1_detector_frame_Msun']
m2_array = data['m2_detector_frame_Msun']
l1_array = data['lambda1']
l2_array = data['lambda2']

training_data = os.listdir("DU17_Training")
fname = "DU17_Training/" + training_data[0]
print(f'file: {fname}')

data = pd.read_pickle(fname)
m1 = np.hstack(data['m1'])
m2 = np.hstack(data['m2'])
l1 = np.hstack(data['l1'])
l2 = np.hstack(data['l2'])

"""plt.subplot(221)
plt.title("m1")
plt.hist(m1_array,bins = 100,label = "170817",alpha = 0.5,density = True)
plt.hist(m1,bins = 100,label = "training data", alpha = 0.5,density = True)
plt.hist(final_dat2[:,0],bins = 100,label = "NS-NS data", alpha = 0.5,density = True)
#plt.vlines(np.mean(m1_array),0,1,label = "Average Used",color = "black")
plt.legend()

plt.subplot(222)
plt.title("m2")
plt.hist(m2_array,bins = 100,label = "170817",alpha = 0.5,density = True)
plt.hist(m2,bins = 100,label = "training data", alpha = 0.5,density = True)
plt.hist(final_dat2[:,1],bins = 100,label = "NS-NS data", alpha = 0.5,density = True)

#plt.vlines(np.mean(m2_array),0,1,label = "Average Used",color = "black")
plt.legend()

plt.subplot(223)
plt.title("Λ1")
plt.hist(l1_array,bins = 100,label = "170817",alpha = 0.5)#,density = True)
plt.hist(l1,bins = 100,label = "training data", alpha = 0.5)#,density = True)
plt.hist(final_dat2[:,2],bins = 100,label = "NS-NS data", alpha = 0.5)#,density = True)
#plt.vlines(np.mean(l1_array),0,0.01,label = "Average Used",color = "black")
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.subplot(224)
plt.title("Λ2")
plt.hist(l2_array,bins = 100,label = "170817",alpha = 0.5)#,density = True)
plt.hist(l2,bins = 100,label = "training data", alpha = 0.5)#,density = True)
plt.hist(final_dat2[:,3],bins = 100,label = "NS-NS data", alpha = 0.5)#,density = True)
#plt.vlines(np.mean(l2_array),0,0.01,label = "Average Used",color = "black")
plt.xscale('log')
plt.yscale('log')
plt.legend()

plt.show()
"""

mm = np.subtract(m1,m2)
ll = np.subtract(l1,l2)
mm2 = np.subtract(m1_array,m2_array)
ll2 = np.subtract(l1_array,l2_array)
mm3 = np.subtract(final_dat2[:,0],final_dat2[:,1])
ll3 = np.subtract(final_dat2[:,2], final_dat2[:,3])


#plt.plot(mm2,ll2,",",label = "170817")
plt.plot(mm,ll,",",label = "training data")
#plt.plot(mm3,ll3,",",label = "NS-NS")
plt.legend()
#plt.yscale('log')
#plt.xscale('log')
plt.xlabel("m1 - m2")
plt.ylabel("Λ1 - Λ2")
plt.show()
