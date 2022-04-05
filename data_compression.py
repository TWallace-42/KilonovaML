import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.utils import shuffle
import random
import time
from DU17_Model import Generate_LightCurve

def compress_file(fname,new_fname,f = 10,troubleshooting = True,zero = False):
    data = pd.read_pickle(fname)

    m1 = data['m1']
    m2 = data['m2']
    l1 = data['l1']
    l2 = data['l2']
    g = data['g'].values
    
    conditionals = np.vstack((m1,m2,l1,l2)).T
    m1 = np.vstack(data['m1'])
    m2 = np.vstack(data['m2'])
    l1 = np.vstack(data['l1'])
    l2 = np.vstack(data['l2'])
    
    band_data = []
    for band in ['g','r','i','z']:
        curve = data[band].values
        curve = np.vstack(curve)
        if zero == True:
            curve = np.nan_to_num(curve)
            
        t_d = data['time']
        t_d = np.vstack(t_d)[0]

        compressed_curve,compressed_t_d = compress_curve(t_d,curve,f)
        band_data.append(compressed_curve)
    
    #compressed_curve = np.array(compressed_curve)
    
    
    band_data = np.array(band_data)
    print(band_data)
    
    final_data = []
    for i in np.arange(len(band_data[0])):
        
        m1,m2,l1,l2 = conditionals[i]
        
        g = band_data[0][i]
        r = band_data[1][i]
        I = band_data[2][i]
        z = band_data[3][i]
        d = np.array([m1,m2,l1,l2,compressed_t_d,g,r,I,z])
        final_data.append(d)
    
    print('check 1')
    final_data = np.array(final_data)
    #print(final_data.shape)
    print('check 2')
    df = pd.DataFrame(data = final_data,
                          columns = list(['m1','m2','l1','l2','time','g','r','i','z']))
    print('check 3')

    
    if zero == True:
        new_fname += "_zero"
    df.to_pickle(f'Data_Cache/Comp/{new_fname}_comp.pkl')
    print(fname," compressed")
    
def compress_curve(t_d,curve,f):
    L = len(curve[0])

    compressed_curve = np.zeros((1,len(curve))).T
    compressed_t_d = []

    for i in np.arange(L):
        if i % f == 0:
            new_column = np.array([np.array(curve[:,i])]).T
            new_t = t_d[i]
            compressed_curve = np.hstack((compressed_curve,new_column))
            compressed_t_d.append(new_t)

    compressed_curve = compressed_curve[:,1:]
    compressed_t_d = np.array(compressed_t_d)
    return(compressed_curve,compressed_t_d)

def crop_index(data,crop_range,searching = False, band = "g"):
    fails = 0
    g = data[band].values
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    nonzero_g = []
    zero_g = []
    
    for i in np.arange(len(g)):
        if np.isnan(g[i]).all():
            fails += 1
            zero_g.append(g[i])
        else:
            nonzero_g.append(g[i])
    #g = []
    nonzero_g = np.array(nonzero_g)
    zero_g = np.array(zero_g)
    print(f'{fails} were empty')

    output = []
    cond_output = []
    if searching == True:
        
        crop_range = len(nonzero_g[0]) - 1
        N = []
        while crop_range >= 10:
            time.sleep(15)
            n = []
            bottom = 0
            top = bottom + crop_range
            
            while top <= len(nonzero_g[0]) - 1:
                n_ = 0
                cropped_temp = nonzero_g[:,bottom:top]
                for g in cropped_temp:
                    if np.isnan(np.min(g)):
                        pass
                    else:
                        n_ += 1
                bottom += 1
                top += 1
                n.append(n_)
            N.append([crop_range,round(100*max(n)/len(nonzero_g),2),n.index(max(n))])
            crop_range -= 50
            print(N[-1])
        N = np.array(N)
        print(N)
    bottom = 0
    top = bottom + crop_range
    n = []
    while top <= len(nonzero_g[0]) - 1:
            n_ = 0
            cropped_temp = nonzero_g[:,bottom:top]
            for g in cropped_temp:
                if np.isnan(np.min(g)):
                    pass
                else:
                    n_ += 1
            bottom += 1
            top += 1
            n.append(n_)
    
    bottom = n.index(max(n))
    top = bottom + crop_range
    cropped_temp = nonzero_g[:,bottom:top]
    cropped_zero = zero_g[:,bottom:top]
    return(bottom,top)
    
def crop_data(fname,crop_range,searching = False,band = 'g',zeroes = False,
              troubleshooting = False,static_bottom = False,init_bot = 0):
    fails = 0
    data = pd.read_pickle(fname)
    g = data[band].values
    t_d = data['time']
    t_d = np.vstack(t_d)[0]
    
    nonzero_g = []
    indices = []
    zero_g = []
    M1 = []
    M2 = []
    L1 = []
    L2 = []
    
    M01 = []
    M02 = []
    L01 = []
    L02 = []
    
    for i in np.arange(len(g)):
        if np.isnan(g[i]).all():
            fails += 1
            zero_g.append(g[i])
            
            m1 = data['m1'].values[i]
            m2 = data['m2'].values[i]
            l1 = data['l1'].values[i]
            l2 = data['l2'].values[i]
            
            M01.append(m1)
            M02.append(m1)
            L01.append(l1)
            L02.append(l2)
        else:
            nonzero_g.append(g[i])
            indices.append(i)
            m1 = data['m1'].values[i]
            m2 = data['m2'].values[i]
            l1 = data['l1'].values[i]
            l2 = data['l2'].values[i]
            
            M1.append(m1)
            M2.append(m1)
            L1.append(l1)
            L2.append(l2)
    #g = []
    nonzero_g = np.array(nonzero_g)
    zero_g = np.array(zero_g)
    print(f'{fails} were empty')

    output = []
    cond_output = []

    """while troubleshooting ==True:
        #PASS
        i = random.randint(0,len(nonzero_g))
        g = data[band].values
        plt.plot(t_d,nonzero_g[i],label = "non-zero_g")
        #print(indices[i])
        plt.plot(t_d,g[indices[i]],":",label = "Original")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()"""

    
    if searching == True:
        
        crop_range = len(nonzero_g[0]) - 1
        N = []
        while crop_range >= 1:
            time.sleep(15)
            
            
            if static_bottom == True:
                #static bottom
                bottom = init_bot
                top = bottom + crop_range
                n = 0
                cropped_temp = nonzero_g[:,bottom:top]
                for g in cropped_temp:
                    if np.isnan(np.sum(g)):
                        pass
                    else:
                        n += 1
                N.append([crop_range,round(100*n/len(nonzero_g),2)])
            else:
                #moving bottom
                n = []
                bottom = 0
                top = bottom + crop_range
                while top <= len(nonzero_g[0]) - 1:
                    n_ = 0
                    cropped_temp = nonzero_g[:,bottom:top]
                    for g in cropped_temp:
                        if np.isnan(np.sum(g)):
                            pass
                        else:
                            n_ += 1
                    bottom += 1
                    top += 1
                    n.append(n_)
                N.append([crop_range,round(100*max(n)/len(nonzero_g),2),n.index(max(n))])
            
            #CHANGE RATE CROP_RANGE GOES DOWN HERE
            crop_range -= 1
            #------------------------------------#

            
            print(N[-1])
        N = np.array(N)
        print(N)
        plt.plot(N[0][:],N[1][:])
        plt.show()

    #g-band OG data: 450
    
    
    
    n = []

    if static_bottom == True:
        bottom = init_bot
        top = bottom + crop_range
        #static bottom
        n_ = 0
        cropped_temp = nonzero_g[:,bottom:top]
        for g in cropped_temp:
            if np.isnan(np.sum(g)):
                pass
            else:
                n_ += 1
        bottom = init_bot
    else:
        bottom = 0
        top = bottom + crop_range
        #Mobile bottom
        while top <= len(nonzero_g[0]) - 1:
                n_ = 0
                cropped_temp = nonzero_g[:,bottom:top]
                for g in cropped_temp:
                    if np.isnan(np.sum(g)):
                        pass
                    else:
                        n_ += 1
                bottom += 1
                top += 1
                n.append(n_)

        bottom = n.index(max(n))

    
    
    top = bottom + crop_range
    cropped_temp = nonzero_g[:,bottom:top]
    cropped_zero = zero_g[:,bottom:top]

    indices2 = []
    nans = 0
    for i in np.arange(len(cropped_temp)):
        if np.isnan(np.sum(cropped_temp[i])):
            nans += 1
        else:
            cond_output.append([M1[i],M2[i],L1[i],L2[i]])
            indices2.append(indices[i])
            output.append(cropped_temp[i])
    print(f'{len(cropped_temp)} - {nans} = {len(output)}')       
    """if troubleshooting == True:
        #PASS
        g = data[band].values
        while troubleshooting == True:
            i = random.randint(0,len(output))
            plt.plot(t_d[bottom:top],output[i],label = "Cropped")
            plt.plot(t_d,g[indices2[i]],":",label = "Original")
            plt.gca().invert_yaxis()
            plt.legend()
            plt.show()
            print(np.sum(output[i]))"""
            
    if zeroes == True:
        for i in np.arange(len(cropped_zero)):
            cond_output.append([M01[i],M02[i],L01[i],L02[i]])
            output.append(cropped_zero[i])
    print("beginning to save data")
        
    t_d = t_d[bottom:top]
    output = np.array(output)
    t_d = list(t_d)
        
    #print(list(output[0]))
    
    final_data = []
    
    for i in np.arange(len(output)):
        #print(f'{100*i/len(output):.3g}%')
        m1 = data['m1'].values[indices2[i]]
        m2 = data['m2'].values[indices2[i]]
        l1 = data['l1'].values[indices2[i]]
        l2 = data['l2'].values[indices2[i]]
        #print(m1,m2,l1,l2)
        d = np.array([m1,m2,l1,l2,t_d,output[i]])
        
        final_data.append(d)

    print("data prepped")
    final_data = np.array(final_data)
    
    bandindex = ['g','r','i','z'].index(band) + 1
    if troubleshooting == True:
        g = data[band].values
    while troubleshooting == True:
        
        i = random.randint(0,len(final_data))
        m1,m2,l1,l2 = final_data[i][0],final_data[i][1],final_data[i][2],final_data[i][3]
        t_d = final_data[i][4]
        output = final_data[i][5]
        lc = Generate_LightCurve(m1,m2,l1,l2)[1]
        #lc = np.nan_to_num(lc)
        
        plt.plot(lc[0],lc[1][bandindex],label = "Model",c = "red")
        plt.plot(t_d,output,"--",label = "training data")

        t_d = data['time']
        t_d = np.vstack(t_d)[0]
        plt.plot(t_d,g[indices2[i]],":",label = "Original")
        plt.gca().invert_yaxis()
        plt.legend()
        plt.show()
    
    df = pd.DataFrame(data = final_data,
                          columns = list(['m1','m2','l1','l2','time',band]))
    df.to_pickle(f'Data_Cache/Crop/DU17_{band}_cropped_{crop_range}.pkl')
    print("df saved")
    print(df)
    #print(len(final_data))

#g og not compressed: 450 crop_range
#g og compressed: 46 crop_range
#r og compressed: 84
#i og compressed: 84
#z og compressed: 84
"""crop_data("Data_Cache/Comp/Comp_91_Original_Combined_comp.pkl",crop_range = 8,
          searching = False,band = 'g',troubleshooting = False,static_bottom = True,
          init_bot = 1)"""

#For compressing a file
factor = 45.5
f = "Original_Combined"
fname = "Data_Cache/Original/" + f + ".pkl"

compress_file(fname,f'Comp_{factor:.2g}_{f}',f = factor,
              troubleshooting = True,zero = True)

