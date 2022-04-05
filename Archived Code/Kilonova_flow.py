import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from glasflow import RealNVP
import torch
from torch import optim

import random
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from DU17_Model import Generate_LightCurve
plt.style.use('seaborn-colorblind')

data = pd.read_pickle("DU17_training/DU17_0.pkl")
data = shuffle(data)

curve = data['g'].values
curve = np.vstack(curve)
scaling_constant = np.min(curve)
print(scaling_sconstant)
curve = np.nan_to_num(curve)
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

device = 'cpu'
flow = RealNVP(
    n_inputs=901,
    n_transforms =4,
    n_conditional_inputs=4,
    n_neurons=32,
    batch_norm_between_transforms=True)

flow.to(device)
print(f'Created flow and sent to {device}')

optimiser = torch.optim.Adam(flow.parameters())

batch_size =  1000
curve_train,curve_val,conditional_train,conditional_val = train_test_split(
    curve,conditional,test_size = 0.33,shuffle = False)

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

epochs = 100
loss = dict(train=[],val=[])
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
    if not i % 10:
        print(f"Epoch {i} - train: {loss['train'][-1]:.3f}, val: {loss['val'][-1]:.3f}")

flow.eval()
print("Finished training")

plt.plot(loss['train'], label='Train')
plt.plot(loss['val'], label='Val.')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()


def many_samples():
    n = 100
    test_array = []
    for i in np.arange(n):
        temp = random.choice(conditional)
        test_array.append(temp)
    test_array = np.array(test_array)
    conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)

    with torch.no_grad():
        samples = flow.sample(n,conditional = conditional)

    samples = samples.cpu().numpy()
    conditional = conditional.cpu().numpy()
    #print(np.mean(samples,axis = 0))


    for i in np.arange(n):
        plt.plot(t_d,samples[i],",",color = "blue")
    plt.plot(t_d,np.mean(samples,axis = 0),color = "red")
    plt.gca().invert_yaxis()
    plt.show()


n = 2
test_array = []
for i in np.arange(n):
    temp = random.choice(conditional)
    test_array.append(temp)
test_array = np.array(test_array)
conditional = torch.from_numpy(test_array.astype(np.float32)).to(device)

with torch.no_grad():
    samples = flow.sample(n,conditional = conditional)
samples = samples.cpu().numpy()
conditional = conditional.cpu().numpy()

m1,m2,l1,l2 = conditional[0]
lc = Generate_LightCurve(m1,m2,l1,l2)[1]
plt.plot(lc[0],lc[1][1],label = "Model[0]",c = "blue")
plt.plot(t_d,samples[0],".",ms = 0.5,label = "Flow[0]",c = "blue")

m1,m2,l1,l2 = conditional[1]
lc = Generate_LightCurve(m1,m2,l1,l2)[1]
plt.plot(lc[0],lc[1][1],label = "Model[1]",c = "red")
plt.plot(t_d,samples[1],".",ms = 0.5,label = "Flow[1]",c = "red")
#print(lc[1][1])

plt.gca().invert_yaxis()
plt.legend()
plt.show()

