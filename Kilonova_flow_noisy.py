import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
from matplotlib import cm
plt.style.use('seaborn-colorblind')

from glasflow import RealNVP
import torch
from torch import optim

import random
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DU17_Model import Generate_LightCurve


#       HYPER PARAMS           #
#------------------------------#
troubleshooting = False #if True you can enter into trouble shooting loops

#basic hyperparameters for the machine learning
epochs = 2000
learning_rate = 1e-5
test_split = 0.33
batch_size =  1000
patience = 0.1
#for learning scheduling
step_f = 1.
gamma = 0.01

#which band we are training to
band = 'r' 
bandindex = ['g','r','i','z'].index(band) + 1

#idiot variables
axis = 0

#         SETUP 1                   #
#-----------------------------------#
#                                   #
training_data = os.listdir("DU17_Training") #data is stored in this folder but typically is only one file
loss = dict(train=[],val=[],delta=[]) #initialise loss dict

device = torch.device('cuda')#set device as GPU

#                                SCALING                        #
#---------------------------------------------------------------#
#  finding the scaling constant to use to normalise the curves  #

fname = "Data_Cache/New/combined.pkl"
data = pd.read_pickle(fname)
data = shuffle(data)
    
curve = data[band].values
curve = np.vstack(curve)
curve = np.nan_to_num(curve)

scaling_constant = np.min(curve)
print(f'scaling_constant {band}: {scaling_constant}')

#scaling_constant = -14.019296288484181 # for g band, just incase I lose it


#                   SETUP 2             #
#---------------------------------------#

fname = "DU17_Training/" + training_data[0]
print(f'file: {fname}')

data = pd.read_pickle(fname)
data = shuffle(data) #shuffling for fun
    
curve = data[band].values
curve = np.vstack(curve)#more convenient for manipulation

#set the Flow AI
flow = RealNVP(
    n_inputs=len(curve[0]),#based on length of training data
    n_transforms =8,
    n_conditional_inputs=4,
    n_neurons=32,
    batch_norm_between_transforms=True)

#send to GPU
flow.to(device)

#optimiser/scheduler setup
optimiser = torch.optim.Adam(flow.parameters(),lr = learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimiser, step_size=step_f*epochs, gamma=gamma)
print(f'Created flow and sent to {device}')

#Ceck that the curves are correctly normalised
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
print(len(m1)," Data points")
m1 = np.vstack(data['m1'])
m2 = np.vstack(data['m2'])
l1 = np.vstack(data['l1'])
l2 = np.vstack(data['l2'])


#                  CONVERTING DATA TO TENSORS   DO NOT TOUCH               #
#--------------------------------------------------------------------------#
#           I don't fully understand what's happening here so best         #
#                        to just leave it as is, it works                  #

data = []
curve_train,curve_val,conditional_train,conditional_val = train_test_split(
    curve,conditional,test_size = test_split,shuffle = False)

y_train = conditional_train
x_train = curve_train
y_val = conditional_val
x_val = curve_val

x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
train_dataset = torch.utils.data.TensorDataset(x_train_tensor,y_train_tensor)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size= batch_size, shuffle = False
    )

x_val_tensor = torch.from_numpy(x_val.astype(np.float32))
y_val_tensor = torch.from_numpy(y_val.astype(np.float32))
val_dataset = torch.utils.data.TensorDataset(x_val_tensor, y_val_tensor)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=False
)

    
#                  TRAINING               #
#-----------------------------------------#
print("beginning training")
for i in range(epochs):
    flow.train()
    train_loss = 0.0
    for batch in train_loader:
        x,y = batch
        x = x.to(device)
        y = y.to(device)
        optimiser.zero_grad()
        _loss = -flow.log_prob(x,conditional = y).mean()
        _loss.backward()
            
        optimiser.step()
            
        train_loss += _loss.item()
    loss['train'].append(train_loss/len(train_loader))

    flow.eval()
    val_loss = 0.0
    for batch in val_loader:
        x,y, = batch
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _loss = -flow.log_prob(x,conditional=y).mean().item()
            
        val_loss += _loss
    loss['val'].append(val_loss / len(val_loader))
        
    scheduler.step()

    #Run every 10 epochs
    if not i % 10:
        try:
            print(f"Epoch {i} - train: {loss['train'][-1]:.4g}"+
                    f"\t val: {loss['val'][-1]:.4g}"+
                    f"\tlr: {scheduler.get_last_lr()[0]:.3g}"+
                    f"\tΔLoss: {loss['val'][-11] - loss['val'][-1]:.4g}")
            try:
                loss['delta'].append(loss['val'][-11] - loss['val'][-1])
                delta = loss['delta'][-5:]
                if all(d < patience for d in delta):
                     if all( d > -1*patience for d in delta):
                        print("Early Stopping")
                        break
            except Exception as e:
                pass
                
        except Exception as e:
            print(e)
            print(f"Epoch {i} - train: {loss['train'][-1]:.4g}"+
                    f"\t val: {loss['val'][-1]:.4g}"+
                    f"\tlr: {scheduler.get_last_lr()[0]:.3g}")
            
    if i % 50 == 0:
        #   TESTING THE AI ON REAL DATA #
        #-------------------------------#
        test_array = []
        indices = []
        N = 3 #number of graphs to predict
        for n in np.arange(N):
            j = random.randint(0,len(m1))
            indices.append(j)
            temp = random.choice(conditional)
            test_array.append(temp)
        
        test_array = np.array(test_array)
        
        cond = torch.from_numpy(test_array.astype(np.float32)).to(device)

        Big_Samples = []
        N_Samples = 100
        cond = torch.from_numpy(test_array.astype(np.float32)).to(device)

        #create N samples
        with torch.no_grad():
            for j in np.arange(N_Samples):
                samples  = flow.sample(N,conditional = cond)
                Big_Samples.append(samples)
        for j in np.arange(len(Big_Samples)):
            Big_Samples[j] = Big_Samples[j].cpu().numpy()



        Big_Samples = np.array(Big_Samples)

        final_samples = np.mean(Big_Samples,axis = axis)

        std = np.std(Big_Samples,axis = axis)
        max_lines = final_samples + 3*std #np.max(Big_Samples,axis = 0)
        min_lines = final_samples - 3*std #np.min(Big_Samples,axis = 0)


        cond = cond.cpu().numpy()
        #print(cond)
        cmap =cm.get_cmap('viridis') #so we can set the colour of all the plots
        for n in np.arange(N):
            #print(f"plotting {n}")
            col = cmap(n/N)
            m1_,m2_,l1_,l2_ = cond[n]
            lc = Generate_LightCurve(m1_,m2_,l1_,l2_)[1]
            #print(m1_,m2_,l1_,l2_)
            lc = np.nan_to_num(lc)
            #print(lc[1][bandindex])
            plt.plot(lc[0],lc[1][bandindex],"--",label = f"[{m1_:.3g}, {m2_:.3g}, {l1_:.3g}, {l2_:.3g}]",c = col)
            plt.plot(t_d,scaling_constant*final_samples[n],"-",ms =4,c = col)
            plt.fill_between(t_d,min_lines[n]*scaling_constant,max_lines[n]*scaling_constant,alpha = 0.2,color = col)

        plt.gca().invert_yaxis()
        plt.title(f'iteration {i}')
        plt.legend()
        plt.savefig(f'Model Evolution/iteration {i}.png')
        #plt.show()
        
        plt.clf()
print("Finished training")

#           EVALUATION          #
#-------------------------------#

#Plot the loss graph
flow.eval()
plt.subplot(211)
plt.plot(loss['train'] + np.abs(np.min(loss['train'])), label='Train')
plt.plot(loss['val']+ np.abs(np.min(loss['val'])), label='Val.')
plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss (log(loss + |min|))')
plt.legend()

plt.subplot(212)
plt.plot(loss['train'], label='Train')
plt.plot(loss['val'], label='Val.')
#plt.yscale('log')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

#plot the ΔLoss graph
plt.plot(loss['delta'])
plt.xlabel('Epoch')
plt.ylabel('ΔLoss (log)')
plt.show()

#   TESTING THE AI ON REAL DATA #
#-------------------------------#
test_array = []
indices = []
N = 3 #number of graphs to predict
for n in np.arange(N):
    i = random.randint(0,len(m1))
    indices.append(i)
    temp = random.choice(conditional)
    test_array.append(temp)
test_array = np.array(test_array)
cond = torch.from_numpy(test_array.astype(np.float32)).to(device)

Big_Samples = []
N_Samples = 100
cond = torch.from_numpy(test_array.astype(np.float32)).to(device)

#create N samples
with torch.no_grad():
    for i in np.arange(N_Samples):
        samples  = flow.sample(N,conditional = cond)
        Big_Samples.append(samples)
for i in np.arange(len(Big_Samples)):
    Big_Samples[i] = Big_Samples[i].cpu().numpy()



Big_Samples = np.array(Big_Samples)

final_samples = np.mean(Big_Samples,axis = axis)

std = np.std(Big_Samples,axis = axis)
max_lines = final_samples + 3* std #np.max(Big_Samples,axis = 0)
min_lines = final_samples - 3* std #np.min(Big_Samples,axis = 0)


cond = cond.cpu().numpy()

cmap =cm.get_cmap('viridis') #so we can set the colour of all the plots
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

#           SAVE THE MODEL      #
#-------------------------------#
models = os.listdir("Models/")
print(models)
i = 0
for m in models:
    if m == f'model_{i}_{band}.pth' :
        print(m," Already taken")
        i += 1
    else:
        torch.save(flow,f"Models/model_{i}_{band}.pth")
        print(f"Model saved as Models/model_{i}_{band}.pth")
torch.save(flow,f"Models/model_{i}_{band}.pth")
print(f"Model saved as Models/model_{i}_{band}.pth")
