import numpy as np
import pandas as pd
import os

#combines data files in the folder fname
fname = "Data_Cache/"
files = os.listdir(fname)
print(files)

objs = []
for f in files:
    data = pd.read_pickle(fname + f)
    objs.append(data)

final = pd.concat(objs)

print(final)
final.to_pickle(fname + "combined.pkl")
