"""
This program solves the stochastic growth model with logarithmic utility 
and Markovian shocks using value function iteration 
and a continuos state space of capital. Iterations performed with loops. 
The Value function is linearly interpolated for any value of the technology shock.

Translated from Eva Carceles-Poveda's MATLAB codes
"""


# Importing packages
import numpy as np
import matplotlib.pyplot as plt

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)


# This is used for the fminbound() function
from scipy import optimize

# This is used to calculate the excution time of the while loop
import time

# This is needed for the function random(), 
# which returns a random floating point number in the range [0.0, 1.0)
import random

# you can either import all the functions defined in the Markov file
# from Markov import *

# or only import the specific functions you want to use
from Markov import markov_approx

# import the maxlininterpol() function
from maxlininterpol import maxlininterpol


# Model Parameters
delta = 0.1
alfa = 0.33
A = 1
beta = 0.9

# Algorithm parameters
tolv = 1e-5 #tolerance, equals 1*10^-7
simyes = 1 # simulates model
stationary = 1 # finds stationary distribution for capital


# Define the shock
rho = 0.95
sigmae = 0.00712
N = 7
m = 3
Pi, teta, P, arho, asigma = markov_approx(rho,sigmae,m,N)
teta = np.exp(teta)

#P = np.linalg.matrix_power(Pi, 10000) # stationary distribution

lt=teta.size


# Grid for capital
lk = int(input('Enter the number of grid points for the capital: '))
k_min = 1
k_max = 3
grk = (k_max-k_min)/(lk-1)
k = np.arange(k_min,k_max+grk,grk)
#k = np.linspace(k_min,k_max,lk).T


# Calculate disposable income (c+k)
yd = np.zeros((lk,lt))

for j in np.arange(0,lt):
    for i in np.arange(0,lk):
        yd[i,j] = teta[j]*k[i]**alfa+(1-delta)*k[i]


# Initialization of the value function
V0 = np.ones((lk,lt))
V1 = np.zeros((lk,lt))
kpr = np.zeros((lk,lt))

iteration = 0
start = time.perf_counter()
while np.abs(V1-V0).max(axis=0).max() > tolv:
    V0 = V1.copy()
    for j in np.arange(0,lt):
        for i in np.arange(0,lk):
            inc = yd[i,j]
            kpr[i,j]=optimize.fminbound(func=maxlininterpol,x1=k[0],\
                                        x2=min(yd[i,j]-0.00001,k[lk-1]),\
                                            args=(V0,beta,inc,Pi,j,k,),xtol=1e-04)
            #print("kpr=",kpr[i,j])
            V1[i,j]=-maxlininterpol(kpr[i,j],V0,beta,inc,Pi,j,k)
    iteration = iteration + 1
    print("iteration =", iteration, ", max(V1-V0) = ", \
          np.abs((V1-V0)).max(axis=0).max())


# Compute the optimal conusmption policy
polc=A*np.dot(teta[:,np.newaxis],np.ones((1,lk))).T*\
    (np.dot(k[:,np.newaxis]**(alfa),np.ones((1,lt)))) \
        + (1-delta)*np.dot(k[:,np.newaxis],np.ones((1,lt))) - kpr

stop = time.perf_counter()

print("Elapsed time in seconds for the while loop is:", round(stop - start,4))


#print(kpr)
#print(polc)


# Find the stationary distribution
if lk < 10:
    stationary = 0


if stationary:
    i = 0
    while kpr[i,0] >= k[i]:
        i = i + 1
        if i >= lk:
            break

    khigh=k[i-1]

    j=0;
    while kpr[j,lt-1] >= k[j]:
        j = j + 1
        if j >= lk:
            break
    
    klow=k[j-1]
    
    print('The lower support of the stationary distribution is', np.around(khigh,4))
    print('The upper support of the stationary distribution is', np.around(klow,4))



# Make plots of value and policy functions
fig, ax = plt.subplots()
ax.plot(k, V1[:,0], 'b')
ax.plot(k, V1[:,-1], 'r--')

ax.set(xlabel='Capital Stock', title='Value Functions')
ax.grid()

fig.savefig("plot_value.jpg", dpi=800)
plt.show()


fig2, axs = plt.subplots(2, 1)

axs[0].plot(k, kpr[:,0], 'b')
axs[0].plot(k, kpr[:,-1], 'r--')
axs[0].plot(k, k, 'y--')
axs[0].set(xlabel='Capital Stock', title='Capital Policy Functions')
axs[0].grid()


axs[1].plot(k, polc[:,0], 'b')
axs[1].plot(k, polc[:,-1], 'r--')
axs[1].set(xlabel='Capital Stock', title='Consumption Policy Functions')
axs[1].grid()


plt.tight_layout()
plt.savefig('plot_policy.jpg', dpi=800)
plt.show()
plt.close(fig2)


# Simulation
if simyes == 1:
    T = int(input('Enter the number of periods for simulation: '))
    
    ktime = np.zeros((T+1,))
    ctime = np.zeros((T,))
    intime = np.zeros((T,))
    tetatime = np.zeros((T+1,))
    tind = np.zeros((T+1,), dtype=int)
    
    tind[0] = 3
    kind = int(lk/2)-1
    tetatime[0] = teta[tind[0]]
    ktime[0]=k[kind]
    
    for ttime in np.arange(0,T):
        inc = tetatime[ttime]*ktime[ttime]**alfa + (1-delta)*ktime[ttime]
        ktime[ttime+1]=optimize.fminbound(func=maxlininterpol,x1=k[0],\
                                          x2=min(inc-0.00001,k[lk-1]),\
                                        args=(V0,beta,inc,Pi,tind[ttime],k,),xtol=1e-04)
               
        ctime[ttime] = inc - ktime[ttime+1]
        intime[ttime] = ktime[ttime+1] - (1-delta)*ktime[ttime]
        shock = random.random()
        i = 0
        while Pi[tind[ttime],0:i+1].sum() < shock:
            i = i + 1
        
        tind[ttime+1] = i
        tetatime[ttime+1] = teta[i]
                       
    outime = tetatime*ktime**alfa                    

    #print('ktime=\n',ktime)
    #print('outime=\n',outime)
    #print('intime=\n',intime)
    #print('ctime=\n',ctime)
    #print('tetatime=\n',tetatime)
    #print('tind=\n',tind)


    ttime = np.arange(0,T)
    
    # Make plots of value and policy functions
    fig3, ax3 = plt.subplots()
    ax3.plot(ttime, outime[1:T+1], 'b',label='=Y')
    ax3.plot(ttime, intime, 'r',label='=I')
    ax3.plot(ttime, ctime, 'g',label='=C')
    ax3.plot(ttime, ktime[1:T+1], 'y',label='=K')

    ax3.set(xlabel='Time', title='Simulation of all variables')
    ax3.grid()
    ax3.legend()

    fig3.savefig("plot_simul.jpg", dpi=800)
    plt.show()


    X = np.concatenate((outime[0:T][:,np.newaxis], intime[:,np.newaxis], \
                        ctime[:,np.newaxis], ktime[0:T][:,np.newaxis]),axis=1)
    #print(X)
    #print(X.shape)
    
    # Some summary statistics
    print('Mean of output, investment, consumption, and capital')
    meanX=np.mean(X,axis=0)
    print(meanX)
    print('Volatility of output, investment, consumption, and capital')
    vol=np.std(X,axis=0)/np.mean(X,axis=0)
    print(vol)



