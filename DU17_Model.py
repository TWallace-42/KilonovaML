import numpy as np
from gwemlightcurves.KNModels import table
import gwemlightcurves.EjectaFits.DiUj2017 as du
from gwemlightcurves.KNModels.io.DiUj2017 import calc_lc
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import random
import sys
from threading import Thread
import h5py
import pandas as pd

#values
#m1,m2,c1,c2 = [1.35,1.35,0.174,0.174] # SLy(1.35,1.35) #163
#m1,m2,c1,c2 = [1.3,1.4,0.142,0.153]

def Generate_LightCurve(m1,m2,l1,l2,plot = False):
    tini = 0
    tmax = 11
    dt = 0.01
    kappa = 10
    eps = 1.58e10
    alp = 1.2
    eth = 0.5

    vmin = 0.02

    #calculate c1 and c2 from l1,l2
    c1 = table.CLove(l1) #Compactness-Love relation for neutron stars
    c2 = table.CLove(l2)
    
    #calculate M_ej
    mb1 = table.EOSfit(m1,c1)
    mb2 = table.EOSfit(m2,c2)
    mej = du.calc_meje(m1,mb1,c1,m2,mb2,c2)

    #calculate v_ej
    v_rho = du.calc_vrho(m1,c1,m2,c2)
    v_z = du.calc_vz(m1,c1,m2,c2)
    vej = du.calc_vej(m1,c1,m2,c2)

    #calculate angles
    th = du.calc_qej(m1,c1,m2,c2)
    ph = du.calc_phej(m1,c2,m2,c2)

    t_d, lbol_d,mag_new = calc_lc(tini,tmax,dt,mej,vej,vmin,th,ph,kappa,eps,alp,eth,flgbct = False)

    #print(mag_new[0])
    if plot == True:

        u = mag_new[0]
        g = mag_new[1]
        r = mag_new[2]
        i = mag_new[3]
        z = mag_new[4]
        y = mag_new[5]
        J = mag_new[6]
        H = mag_new[7]
        K = mag_new[8]

        plt.style.use("bmh")
        plt.subplot(121)
        plt.plot(t_d,u,label = "u")
        plt.plot(t_d,g,label = "g")
        plt.plot(t_d,r,label = "r")
        plt.plot(t_d,i,label = "i")
        plt.plot(t_d,z,label = "z")
        #plt.xscale("log")
        plt.xticks(np.arange(tini,tmax))
        plt.gca().invert_yaxis()
        plt.legend(prop={'size': 6})

        plt.subplot(122)
        #plt.plot(t_d,y,label = "y")
        plt.plot(t_d,J,label = "J")
        plt.plot(t_d,H,label = "H")
        plt.plot(t_d,K,label = "K")
        plt.xticks(np.arange(tini,tmax))
        plt.gca().invert_yaxis()
        plt.legend(prop={'size': 6})
        #plt.xscale("log")
        plt.show()
    return([m1,m2,l1,l1],np.array([t_d,mag_new]))



def generate_data(data):
    output = []
    for i in np.arange(len(data)):
        line = data[i]
        if "idlelib" not in sys.modules:
            print(f'\r{100*i/len(data):.3f}% finished',end = '\r')
        m1,m2,l1,l2 = line
        temp_in,temp_out = Generate_LightCurve(m1,m2,l1,l2)
        output.append([temp_in,temp_out])
    return(output)




def thread_fn(m1,m2,l1,l2,fname,printing = False):
    final_data = []
    L = len(m1)
    for i in np.arange(L):
        if printing == True:
            if "idlelib" not in sys.modules:
                print(f'\r{100*i/L:.3f}% finished',end = '\r')
            else:
                if not i % 1000:
                    print(f'{i}/{L}\t{100*i/L:.2f}%')

        for x in np.arange(1):
            m1_ = m1[i] #+ random.uniform(0,0.01)*m1[i] #m1 > m2 => l1 < l2
            m2_ = m2[i] #+ random.uniform(-0.01,0)*m2[i]
            l1_ = l1[i] #+ random.uniform(-0.01,0)*l1[i]
            l2_ = l2[i] #+ random.uniform(0,0.01)*l2[i]
                
            conditions,lightcurves = Generate_LightCurve(m1[i],m2[i],l1[i],l2[i])
            t_d,curves = lightcurves
            m1_,m2_,l1_,l2_ = conditions
            g = curves[1]
            r = curves[2]
            I = curves[3]
            z = curves[4]

            d = np.array([m1_,m2_,l1_,l2_,t_d,g,r,I,z])
            final_data.append(d)

    final_data = np.array(final_data)
    df = pd.DataFrame(data = final_data,
                          columns = list(['m1','m2','l1','l2','time','g','r','i','z']))

    df.to_pickle(f"{fname}.pkl")
    print(f"{fname} done")

def thread_fn2(fname,i,printing = False):
    print(f'thread {i} starting')
    final_data = []
    data = np.array(pd.read_pickle(fname).values)
    t = 0
    
    for d in data:
        L = len(data)
        t += 1
        if printing == True:
            if "idlelib" not in sys.modules:
                print(f'\r{100*t/L:.3f}% finished',end = '\r')
            else:
                if not i % 1000:
                    print(f'{t}/{L}\t{100*t/L:.2f}%')
        m1,m2,l1,l2,t_d,g,r,I,z = d
        for j in np.arange(5):
            m1_ = m1 + random.uniform(0,0.01)*m1 #m1 > m2 => l1 < l2
            m2_ = m2 + random.uniform(-0.01,0)*m2
            l1_ = l1 + random.uniform(-0.01,0)*l1
            l2_ = l2 + random.uniform(0,0.01)*l2
            new_d = np.array([m1_,m2_,l1_,l2_,t_d,g,r,I,z])
            final_data.append(new_d)
            
    final_data = np.array(final_data)

    #print(final_data[0])
    df = pd.DataFrame(data = final_data,
                      columns = list(['m1','m2','l1','l2','time','g','r','i','z']))
    df.to_pickle(f"{fname}_noise.pkl")
    print(f'thread {i} finished')
    
#testing Jordan's code
if __name__ == "__main__":
    
    """#adding noise
    N_threads = 16
    threads = []
    
    for i in np.arange(N_threads):
        printing = False
        if i == 16: #16 so there is no progress %
            printing = True
        
        fname = f"DU17_training/DU17_{i}.pkl"
        x = Thread(target = thread_fn2, args = (fname,i,printing,))
        threads.append(x)
        x.start()
    
    for thread in threads:
        thread.join()
    print("All threads finished")"""
    
    #Making the first data
    filedir = "mass_lambda/mass_lambda_distributions.h5"

    d = h5py.File(filedir, 'r')
    data = np.array(d.get('labels'))
    d.close()

    m1 = data[:,0]
    m2 = data[:,1]
    l1 = np.exp(data[:,2])
    l2 = np.exp(data[:,3])

    N_threads = 16
    
    part_m1 = np.split(m1, N_threads)
    part_m2 = np.split(m2, N_threads)
    part_l1 = np.split(l1, N_threads)
    part_l2 = np.split(l2, N_threads)
    threads = list()
    
    for i in np.arange(N_threads):
        printing = False
        if i == 0:
            printing = True
        x = Thread(target = thread_fn, args = (part_m1[i],part_m2[i],part_l1[i],part_l2[i],
                                                         f'DU17_training/DU17_{i}',printing,))
        threads.append(x)
        x.start()
        
    for thread in threads:
        thread.join()
    print("All threads finished")

    #print('m1\tm2\tl1\tl2')
    

    
    
