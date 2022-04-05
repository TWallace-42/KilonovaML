import numpy as np
import matplotlib.pyplot as plt
import random 
def distribution(func,range_,*args):
    x = np.linspace(range_[0],range_[1],100)
    y = func(x,*args)

    choice = random.choices(x,y)
    return(choice)
    #plt.plot(x,y)
    #plt.show()
    
def gaussian(x,a,b,c):
    A = (x-b)**2
    B = -A/(2*c**2)

    f = a*np.exp(B)
    return(f)

def two_gaussian(x,a1,b1,c1,a2,b2,c2):
    f1 = gaussian(x,a1,b1,c1)
    f2 = gaussian(x,a2,b2,c2)
    f = f1+f2
    return(f)

"""#verifying it works
array = []
for x in range(100000):

    choice = distribution(two_gaussian,[-10,10],1,0,1,2,6,2)
    array.append(choice)

print("plotting")

unique, counts = np.unique(array, return_counts=True)
counts = counts/max(counts)
new_array = np.asarray((unique, counts)).T
x = np.linspace(-10,10,10000)
y = two_gaussian(x,1,0,1,2,6,2)
y = y/max(y)
for element in new_array:
    plt.vlines(element[0],0,element[1],color = "blue")
#plt.plot(unique,counts,".")
plt.plot(x,y,c = "red")
plt.show()
print(np.mean(array))"""
