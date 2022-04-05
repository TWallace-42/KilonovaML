from gwemlightcurves.KNModels import table
from gwemlightcurves.EjectaFits import DiUj2017 as du
import matplotlib
matplotlib.use("TkAgg")
import numpy as np
import matplotlib.pyplot as plt


plt.style.use("bmh")
#Global variables
M_solar = 1.988e30
L_solar = 3.846e33 #e26 Watts e33 ergs per second
c = 3e8
kappa = 10
epsilon = 1.58e10
alpha = 1.2
eth = 0.5


#BC factors Eqn (22a-f) DU17 paper
a_z = [[1.072,0.3646,-0.1032,0.00368,-0.0000126],15] #second number is the cutoff
a_i = [[0.6441,0.0796,-0.122,0.00793,-0.000122],15]
a_r = [[-2.308,1.445,-0.5740,0.0531,-0.00152],15]#Coefficients for r-band Bolometric Corrections
a_g = [[-6.195,4.054,-1.754,0.2246,-0.009813],8.5]
a_u = [[40.01,-56.79,25.73,-5.207,0.3813],5]
a_k = [[-7.876,3.245,-0.3946,0.0216,-0.000443],15]
a_h = [[-2.763,1.502,-0.2133,0.0128,-0.000288],15]
a_j = [[-1.038,1.348,-0.2364,0.0137,-0.000261],15]

a_array = [a_z,a_i,a_r,a_g,a_u,a_k,a_h,a_j]

#initial parameters
m1,m2,c1,c2 = [1.35,1.35,0.174,0.174] # SLy(1.35,1.35) #163
m1,m2,c1,c2 = [1.3,1.4,0.142,0.153] # H4(1.3,1.4) #40



def calc_tc(theta_ej,kappa,M_ej,phi_ej,v_ej,v_min = 0.02):
    numerator = theta_ej*kappa*M_ej

    v_max = 2*v_ej -v_min
    denomenator = 2*phi_ej*(v_max-v_min)
    temp = numerator/denomenator
    return(np.sqrt(temp))



def L_bol(t,theta_ej,e_th,e_0,M_ej,t_c,alpha):
    "Returns Bolometric Luminosity for a lightcurve, Eqn (17) [1]"
    # (1+θ_ej)Ɛ_th* d/dt(Ɛ_0) * M_ej * { (t/t_c)(t/1day)**(-α), t<= t_c [1]
    #                                  { (t/1day)**(-α), t > t_c
    
    #define a day
    d = 1#24*60*60
    
    #split the equations into more manageable parts
    A = (1+theta_ej)*e_th*e_0*M_ej*M_solar
    
    if t<=t_c:
        B = (t/t_c)*(t)**(-alpha)
    else:
        #print("here")
        B = (t)**(-alpha)
    #print(A*B)
    return(A*B)

def M_bol(L):
    "Returns Bolometric Magnitude, Eqn (18) [1]"
    M = 4.74 - 2.5 * np.log10(L/L_solar) #relation from [1]
    return(M)


def M_X(t,M1,M2,C1,C2,a_x,M_ej = None):
    "Magnitude spectrum of band X, Eqn (20) [1]"
    
    #baryonic mass
    mb1 = table.EOSfit(m1,c1)
    mb2 = table.EOSfit(m2,c2)
    #print("\n",mb1,mb2,"\n")
    #ejecta mass
    if M_ej == None:
        M_ej = du.calc_meje(m1,mb1,c1,m2,mb2,c2) #0.7e-3*M_solar

    #velocities
    v_rho = du.calc_vrho(m1,c1,m2,c2)
    v_z = du.calc_vz(m1,c1,m2,c2)
    v_ej = du.calc_vej(m1,c1,m2,c2)

    theta_ej = du.calc_qej(m1,c1,m2,c2)
    phi_ej = du.calc_phej(m1,c2,m2,c2)
    
    #print(f'M_ej: {M_ej}\nv_rho: {v_rho}\nv_z: {v_z}\n,v_ej: {v_ej}')
    
    t_c = calc_tc(theta_ej,kappa,M_ej,phi_ej,v_ej)
    #print(f'{t_c:.3g}')

    L = L_bol(t,theta_ej,eth,epsilon,M_ej,t_c,alpha)
    
    M_x = M_bol(L) - BC_X(t,a_x,M_ej)
    return(M_x)

def BC_X(t,a_x,M_ej):
    "A function which returns the bolometric correction for the bandwidth M_x, Eqn (21) [1]"
    f = ((10e-2)/M_ej)**(1/3.2) #the ratio to convert between t and t' as given in [1]
    a0,a1,a2,a3,a4 = a_x[0]
    t_ = f*t
    BC = a0 + a1*(t_) + a2*(t_)**2 + a3*(t_)**3 + a4*(t_)**4
    return(BC)

#--------Plotting----------#
def plot_BC(M_ej):
    labels = ["z","i","r","g","u","K","H","J"]
    #print(labels[:-3],labels[-3:])
    M = []
    for a_x in a_array:
        temp = []
        for t in np.linspace(0.01,24,100):
            temp.append(BC_X(t,a_x,M_ej))
        M.append(temp)
    f = ((10e-2)/M_ej)**(1/3.2)
    
    plt.subplot(121)
    plt.xlim(2,14)
    plt.ylim(-12,2)
    plt.xticks(np.arange(2,14))
    for element in M[:-3]:
        
        t = f*np.linspace(0.01,24,100)
        i = M.index(element)
        t_cutoff = a_array[i][1]
        cutoff = np.where(t>=t_cutoff)[0][0]
        
        #print(t_cutoff)
        plt.plot(t[:cutoff],element[:cutoff],label = labels[i])
    plt.legend()
    
    plt.subplot(122)
    plt.xlim(2,14)
    plt.ylim(-3,4)
    for element in M[-3:]:
        i = M.index(element)

        t = f*np.linspace(0.01,24,100)
        i = M.index(element)
        t_cutoff = a_array[i][1]
        cutoff = np.where(t>=t_cutoff)[0][0]
        
        plt.plot(t[:cutoff],element[:cutoff],label = labels[i])
    plt.xticks(np.arange(2,14))
    plt.legend()
    plt.show()

def plot_MX(m1,c1,m2,c2,M_ej = None, show = True, label = None,subplots = [121,122]):
    labels = ["z","i","r","g","u","K","H","J"]
    #print(labels[:-3],labels[-3:])
    M = []
    for a_x in a_array:
        #print(a_x)
        temp = []
        for t in np.linspace(0.01,24,100):
            temp.append(M_X(t,m1,m2,c1,c2,a_x,M_ej))
        M.append(temp)
    f = ((10e-2)/M_ej)**(1/3.2)
    
    plt.subplot(subplots[0])

    for element in M[:-3]:
        
        t = f*np.linspace(0.01,24,100)
        i = M.index(element)
        t_cutoff = a_array[i][1]
        cutoff = np.where(t>=t_cutoff)[0][0]
        t0 = np.where(t>=2)[0][0]
        plt.plot(t[:cutoff],element[:cutoff],label = labels[i])

    plt.xlim(2,11)
    y_min = np.array(M)[:-3,t0:cutoff].min() -1
    y_max = np.array(M)[:-3,t0:cutoff].max() +1
    plt.ylim(y_min,y_max)
    plt.xticks(np.arange(2,11))
    
    plt.legend(prop={'size': 6})
    plt.gca().invert_yaxis()
    
    plt.subplot(subplots[1])
    
    plt.xlim(2,11)
    y_min = np.array(M)[-3:,:cutoff].min() -1
    y_max = np.array(M)[-3:,:cutoff].max() +1
    plt.ylim(y_min,y_max)
    plt.xticks(np.arange(2,11))
    
    for element in M[-3:]:
        i = M.index(element)

        t = f*np.linspace(0.01,24,100)
        i = M.index(element)
        t_cutoff = a_array[i][1]
        cutoff = np.where(t>=t_cutoff)[0][0]
        
        plt.plot(t[:cutoff],element[:cutoff],label = labels[i])
    plt.gca().invert_yaxis()
    plt.legend(prop={'size': 6})
    
    if show == True:
        plt.show()

    
def plot_Lum(m1,c1,m2,c2,M_ej = None,show = True,label = None):
    
     
    #print("\n",mb1,mb2,"\n")
    #ejecta mass
    if M_ej == None:
        mb1 = table.EOSfit(m1,c1)
        mb2 = table.EOSfit(m2,c2)
        M_ej = du.calc_meje(m1,mb1,c1,m2,mb2,c2) #0.7e-3*M_solar

    #velocities
    v_rho = du.calc_vrho(m1,c1,m2,c2)
    v_z = du.calc_vz(m1,c1,m2,c2)
    v_ej = du.calc_vej(m1,c1,m2,c2)

    theta_ej = du.calc_qej(m1,c1,m2,c2)
    phi_ej = du.calc_phej(m1,c2,m2,c2)
    
    #print(f'M_ej: {M_ej}\nv_rho: {v_rho}\nv_z: {v_z}\n,v_ej: {v_ej}')
    
    t_c = calc_tc(theta_ej,kappa,M_ej,phi_ej,v_ej)
    print(f'{t_c:.3g}')
    L = []
    for t in np.linspace(0.01,25,100):
        L.append(L_bol(t,theta_ej,eth,epsilon,M_ej,t_c,alpha))

    t = np.linspace(0.01,25,100)

    
    plt.plot(t,L,label = label)
    plt.legend()
    
    if show == True:
        plt.yscale("log")
        plt.show()





SLy = [1.35,1.35,12.2e-3,0.174,0.174,"SLy(1.35,1.35)"] #[M1,M2,M_ej,C1,C2] #163
H4_0 = [1.3,1.4,0.7e-3,0.142,0.153,"H4(1.30,1.40)"] #40
H4_1 = [1.2,1.5,4.5e-3,0.131,0.164,"H4(1.20,1.40)"] #34
APR4_0 = [1.3,1.4,8e-3,0.170,0.182,"APR4(1.30,1.40)"] #20
APR4_1 = [1.2,1.5,7.5e-3,0.157,0.195,"APR4(1.20,1.50)"]#13,14
data_array = [SLy,H4_0,H4_1,APR4_0,APR4_1]


m1,m2,M_ej,c1,c2,label = SLy
plot_MX(m1,c1,m2,c2,M_ej = M_ej,show = False,label = label,subplots = [221,222])

m1,m2,M_ej,c1,c2,label = H4_1
plot_MX(m1,c1,m2,c2,M_ej= M_ej,show = False, label = label,subplots = [223,224])

plt.show()

"""m1,m2,M_ej,c1,c2,label = SLy
plot_BC(M_ej)"""


"""for d in data_array:
    m1,m2,M_ej,c1,c2,label = d
    plot_Lum(m1,c1,m2,c2,M_ej = M_ej,show = False,label = label)
plt.yscale('log')
plt.show()"""

