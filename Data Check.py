import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#imported the model you want to use
from DU17_Model import Generate_LightCurve


"""A programme to compare the data created by the model and the data you have after processing"""

#setting up matplotlib
plt.style.use('seaborn-colorblind')

#setting up data
band = 'g' #band you are comparing
fname = "Data_Cache/combined_nannum.pkl" #location you want to check
data = pd.read_pickle(fname)

def check_data(band,data):
    bandindex = ['g','r','i','z'].index(band) + 1
    
    curve = data[band].values #get lightcurves from loaded data
    curve = np.vstack(curve)

    #get the conditions
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
    
    ind = random.randint(0,len(curve))#index randomly chosen
    m1_i,m2_i,l1_i,l2_i = conditional[ind]
    curve_i = curve[ind] #curve chosen from same index

    #generate ligthcurves
    lc = Generate_LightCurve(m1_i,m2_i,l1_i,l2_i)[1]
    lc = np.nan_to_num(lc)

    #plot lightcurves from generate lightcurve function
    plt.plot(lc[0],lc[1][bandindex],label = "Model",c = "red")

    #plot loaded data
    plt.plot(t_d,curve_i,label = "training data")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.show()

while True:
    check_data(band,data)
