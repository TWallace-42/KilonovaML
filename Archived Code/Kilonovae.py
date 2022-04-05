import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#test comment

#Global variables
M_solar = 1.988e30
L_solar = 3.846e33 #e26 Watts e33 ergs per second
c = 3e8

#DU17 model variables, Eqn (22a-f) [1]
a_z = [[1.072,0.3646,-0.1032,0.00368,-0.0000126],15] #second number is the cutoff
a_i = [[0.6441,0.0796,-0.122,0.00793,-0.000122],15]
a_r = [[-2.308,1.445,-0.5740,0.0531,-0.00152],15]#Coefficients for r-band Bolometric Corrections
a_g = [[-6.195,4.054,-1.754,0.2246,-0.009813],8.5]
a_u = [[40.01,-56.79,25.73,-5.207,0.3813],5]
a_k = [[-7.876,3.245,-0.3946,0.0216,-0.000443],15]
a_h = [[-2.763,1.502,-0.2133,0.0128,-0.000288],15]
a_j = [[-1.038,1.348,-0.2364,0.0137,-0.000261],15]

a_array = [a_z,a_i,a_r,a_g,a_u,a_k,a_h,a_j]

#DU17 Functions
def L_bol(t,theta_ej,e_th,e_0,M_ej,t_c,alpha):
    "Returns Bolometric Luminosity for a lightcurve, Eqn (17) [1]"
    # (1+θ_ej)Ɛ_th* d/dt(Ɛ_0) * M_ej * { (t/t_c)(t/1day)**(-α), t<= t_c [1]
    #                                  { (t/1day)**(-α), t > t_c
    
    #define a day
    d = 1#24*60*60

    #split the equations into more manageable parts
    A = (1+theta_ej)*e_th*e_0*M_ej
    
    if t<=t_c:
        B = (t/t_c)*(t/d)**(-alpha)
    else:
        #print("here")
        B = (t/d)**(-alpha)
    
    return(A*B)

def M_bol(L):
    "Returns Bolometric Magnitude, Eqn (18) [1]"
    M = 4.74 - 2.5 * np.log10(L/L_solar) #relation from [1]
    return(M)

def BC_X(t,a_x,M_ej):
    "A function which returns the bolometric correction for the bandwidth M_x, Eqn (21) [1]"
    f = ((10e-2)/M_ej)**(1/3.2) #the ratio to convert between t and t' as given in [1]
    a0,a1,a2,a3,a4 = a_x[0]

    BC = a0 + a1*(f*t) + a2*(f*t)**2 + a3*(f*t)**3 + a4*(f*t)**4
    return(BC)

def M_X(t,M1,M2,M_ej,C1,C2, a_x, e_th= 1, e_0= 1.58e13, alpha = 1.3,kappa = 10,return_L = False, v_p = None, v_z = None):
    "Magnitude spectrum of band X, Eqn (20) [1]"

    if v_p == None:
        v_p,v_z = velocities(M1,M2,C1,C2)
    
    v_min = 0.02 #they set it to this in [1]
    v_max = vmax(M1,M2,C1,C2,v_min = v_min,v_p = v_p, v_z = v_z)
    
    theta_ej = theta_ejecta(v_p,v_z)
    
    t_c =  t_critical(theta_ej,kappa,M_ej,v_max,v_min)
    #print(f'{t_c:.3g}')
    
    L = L_bol(t,theta_ej,e_th,e_0,M_ej,t_c,alpha)
    if return_L == True:
        return(L)
    else:
        M_x = M_bol(L) - BC_X(t,a_x,M_ej)
        return(M_x)

def t_critical(theta_ej,kappa, M_ej,v_max,v_min):
    "Eqn (16) of [1] to work out the critical time"
    
    phi_ej = 4*theta_ej + np.pi/2
    numerator = theta_ej*kappa*M_ej*M_solar


    
    delta_v = (v_max - v_min)
    #print(f'{delta_v/c:.3g}')
    denomenator = 2*phi_ej*delta_v*c
    
    t_c = np.sqrt(numerator/denomenator)
    #t_c = t_c/(24*60*60) #put t_c in days
    #print(t_c)
    return(t_c)

def vmax(M1,M2,C1,C2,v_min = 0.02,v_p = None, v_z = None):
    "returns the maximum velocity, defined just above Sect 4.2.2 in [1]"
    v_ej = v_ejecta(M1,M2,C1,C2,v_rho = v_p, v_z = v_z)
    return(2*v_ej - v_min)

def velocities(M1,M2,C1,C2):
    "returns v_rho and v_z"
    #first calculate v_rho ([1] (5))
    
    A = [-0.219479,0.444836,-2.67385] #a,b,c for the v_rho polynomial
    v_rho = A[0]*((M1/M2)*(1+A[2]*C1) + (M2/M1)*(1+A[2]*C2)) + A[1] # + (1<->2)?? idk what that means in (5)
    # C = compactness

    #similairly calculate v_z
    B = [-0.315585,0.63808,-1.00757]
    v_z = (B[0]*(M1/M2)*(1+B[2]*C1)) + (B[0]*(M2/M1)*(1+B[2]*C2)) + B[1]
    
    return(v_rho,v_z)

def v_ejecta(M1,M2,C1,C2,v_rho = None,v_z = None):
    "velocity of the ejected mass accourding to (9) in [1]"
    if v_rho == None:
        v_rho,v_z = velocities(M1,M2,C1,C2)
    #calculate v_ej
    v_ej = np.sqrt(v_rho**2 + v_z**2)
    #print(f'{v_ej:.2g}')
    return(v_ej)

def theta_ejecta(v_p,v_z):
    "The estimate of [1] to get theta_ej"
    a = np.sqrt(9*v_z**2 + 4 * v_p **2) # useful to just do this once and keep equations simple
    numerator = 2**(4/3) * (v_p)**2 - (2**(2/3))*(v_p**2 * (3*v_z + a))**2/3
    denomenator = (v_p**5 * (3 * v_z + a))**(1/3)
    #print(numerator/denomenator)
    return(numerator/denomenator)

# e_th in [2]
# e_0 in [1]
# alpha in [1]

M = []


#-----------------Collision Params for Testing----------------#
SLy = [1.35,1.35,12.2e-3,0.174,0.174,"SLy(1.35,1.35)",None,None] #[M1,M2,M_ej,C1,C2] #163
H4_0 = [1.3,1.4,0.7e-3,0.142,0.153,"H4(1.30,1.40)",0.18,0.1] #40
H4_1 = [1.2,1.5,4.5e-3,0.131,0.164,"H4(1.20,1.40)",0.21,0.09] #34
APR4_0 = [1.3,1.4,8e-3,0.170,0.182,"APR4(1.30,1.40)",0.19,0.12] #20
APR4_1 = [1.2,1.5,7.5e-3,0.157,0.195,"APR4(1.20,1.50)",0.24,0.12]#13,14

Raw_data = [SLy,H4_0,H4_1,APR4_0,APR4_1]
#---------------------------Plotting--------------------------#

"""for d in Raw_data:
    L = []
    M1,M2,M_ej,C1,C2,label,v_p,v_z = d
    f = ((10e-2 * M_solar )/M_ej)**(1/3.2)
    time_space = np.linspace(0.01,25,1000)
    for t in time_space:
        L.append(M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_z,return_L = True,v_p = v_p, v_z = v_z)) # a_x is arbitrary if return_L = Trye

    plt.plot(time_space, L,label = label)


plt.yscale("log")
#plt.xlim(0,25)
#plt.ylim(10e37,10e41)
plt.title("Bolometric Luminosity vs Time")
plt.xlabel("time [days]")
plt.ylabel("L_bol [erg/s]")
plt.legend()
plt.show()"""

M1,M2,M_ej,C1,C2,label,v_p,v_z = SLy
f = ((10e-2)/M_ej)**(1/3.2)

for t in np.linspace(1/1000,11,1000):
    Z = BC_X(t,a_z,M_ej)
    I = BC_X(t,a_i,M_ej)
    R = BC_X(t,a_r,M_ej)
    G = BC_X(t,a_g,M_ej)
    U = BC_X(t,a_u,M_ej)
    K = BC_X(t,a_k,M_ej)
    H = BC_X(t,a_h,M_ej)
    J = BC_X(t,a_j,M_ej)
    M.append([Z,I,R,G,U,K,H,J])



"""for t in np.linspace(1/1000,11,1000):
    Z = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_z)
    I = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_i)
    R = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_r)
    G = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_g)
    U = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_u)
    K = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_k)
    H = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_h)
    J = M_X(t,M1 = M1, M2 = M2,M_ej = M_ej, C1 = C1, C2 = C2, a_x = a_j)
    M.append([Z,I,R,G,U,K,H,J])"""
    #L.append(L_bol(t,theta_ej = 0,e_th = 1,e_0 = 1.58e10,M_ej = 0.1*M_solar, t_c = 10,alpha = 1.3))


M = np.array(M)

labels = [['z',0],['i',1],['r',2],['g',3],['u',4],['k',5],['h',6],['j',7]]

#ugriz bands
ordered = [['u',4],['g',3],['r',2],['i',1],['z',0]]
#JHK bands
ordered2  = [['J',7],['H',6],['K',5]]
data = []

minimum_M = np.mean(M[0])
maximum_M = np.mean(M[0])
plt.subplot(121)
for x in np.arange(len(ordered)):
    #plt.subplot(411 + x)
    
    plt.ylabel("Magnitude")
    plt.xlabel("Time [days]")
    #plt.gca().invert_yaxis()

    #reframe the t coordinate
    t = f*np.linspace(1/1000,11,1000)

    start_t = np.where(t <= 2)[0][-1]
    
    #stop plotting after the cutoff point
    t_cutoff = a_array[ordered[x][1]][1]
    cutoff = np.where(t >= t_cutoff)[0][0]

    temp_min = min(M[start_t:cutoff,int(ordered[x][1])])
    temp_max = max(M[start_t:cutoff,int(ordered[x][1])])

    if temp_min < minimum_M:
        
        minimum_M = temp_min
    if temp_max > maximum_M:
        maximum_M = temp_max

    
    plt.plot(t[:cutoff],M[:cutoff,int(ordered[x][1])],label = ordered[x][0])


plt.xlim(2,11)
plt.ylim(round(minimum_M),round(maximum_M))
#plt.ylim (-12,2)


#plt.plot(np.linspace(1/1000,7,1000),L,label = "luminosity")

#plt.yscale("log")
#plt.xscale("log")


#plt.gca().invert_yaxis()
plt.legend()

plt.subplot(122)
for x in np.arange(len(ordered2)):
    plt.ylabel("Magnitude")
    plt.xlabel("Time [days]")
    #reframe the t coordinate
    t = f*np.linspace(1/1000,11,1000)

    start_t = np.where(t <= 2)[0][-1]
    
    #stop plotting after the cutoff point
    t_cutoff = a_array[ordered2[x][1]][1]
    cutoff = np.where(t >= t_cutoff)[0][0]

    temp_min = min(M[start_t:cutoff,int(ordered2[x][1])])
    temp_max = max(M[start_t:cutoff,int(ordered2[x][1])])

    if temp_min < minimum_M:
        minimum_M = temp_min
    if temp_max > maximum_M:
        maximum_M = temp_max

    plt.plot(t[:cutoff],M[:cutoff,int(ordered2[x][1])],label = ordered2[x][0])
plt.xlim(2,11)
plt.ylim(round(minimum_M),round(maximum_M))
#plt.gca().invert_yaxis()
plt.legend()


plt.tight_layout()

plt.yticks(np.arange(-18,2, 1.0))
plt.xticks(np.arange(2, 16, 1.0))
plt.show()
#----------------------------Loading Data----------------------#
Data=False
if Data == True:
    for f in np.linspace(0.01,1.5,1000):
        M_temp = [] #temporary matrix to hold the magnitude line for the fraction of solar mass
        
        M_ej = f*M_solar

        for t in np.linspace(1/1000,7,1000):
            R = M_X(t,a_r, M_ej = M_ej, theta_ej = 0, e_th = 1, e_0 = 1.58e10, t_c = 10, alpha = 1.3)
            M_temp.append(R)
            
        M_temp = np.array(M_temp)
        M.append(M_temp)

    M = np.array(M)
    mass = np.linspace(0.01,1.5,1000)

    d = []
    for x in np.arange(1000):
        d.append(np.array([mass[x],M[x]]))

    d = np.array(d)
    df = pd.DataFrame(data = d,
                      columns = list(["mass","r-band"]))

    df.to_pickle("r band dataframe.pkl")



#print(df.loc[0,:])
# [1] https://iopscience.iop.org/article/10.1088/1361-6382/aa6bb0/pdf
# [2] https://link.springer.com/content/pdf/10.1007/s41114-017-0006-z.pdf
