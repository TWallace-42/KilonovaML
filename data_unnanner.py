import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import shuffle
import random
import time

from math import isnan


f = 120 #Factor to reduce the data by
fname = "Data_Cache/New/DU17_15.pkl"
print(f'file: {fname}')

data = pd.read_pickle(fname)
m1 = np.hstack(data['m1'])
m2 = np.hstack(data['m2'])
l1 = np.hstack(data['l1'])
l2 = np.hstack(data['l2'])
time = np.vstack(data['time'])[0]

g = np.vstack(data['g'])
print(g.shape)
time = time[::f]
g = g[:,::f]
print(g.shape)
r = np.vstack(data['r'])
r = r[:,::f]
i = np.vstack(data['i'])
i = i[:,::f]
z = np.vstack(data['z'])
z = z[:,::f]

curves = [g,r,i,z]
new_curves = [[],[],[],[]]
#N = len(g)
for i in np.arange(len(curves)):
    print(i)
    #n = 0
    
    for line in curves[i]:
        #print(f'\r{100*n/N}%',end = '\r')
        if np.isnan(line).all():
            new_line = np.nan_to_num(line)
            new_curves[i].append(new_line)
        else:
            first = next(x for x in line if not isnan(x))
            last = next(x for x in line[::-1] if not isnan(x))
            
            split_index = int(round(len(line)/2))
            front,back = np.split(line,2)
            front = np.nan_to_num(front,nan = first)
            back = np.nan_to_num(back,nan = last)
            new_line = np.concatenate((front,back))
            new_curves[i].append(new_line)
        #n+=1
print("done compressing")

new_curves = np.array(new_curves)
g,r,i,z = new_curves

j = random.randint(0,len(g))
final_data = []
for j in np.arange(len(g)):
    d = np.array([m1[j],m2[j],l1[j],l2[j],time,g[j],r[j],i[j],z[j]]) 
    final_data.append(d)

final_data = np.array(final_data)

df = pd.DataFrame(data = final_data,
                  columns = list(['m1','m2','l1','l2','time','g','r','i','z']))

df.to_pickle(f'{fname}_nannum.pkl')
print("done")
