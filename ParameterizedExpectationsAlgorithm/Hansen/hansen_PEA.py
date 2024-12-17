"""
Hansen model with divisible labor by using PEA

This code solves business cycle model with AR(1) shocks 
following Hansen (1985) by using PEA. 
The model includes both labor choice and uncertainty. 
We detrend data by applying the HP filter and 
then compare summary statistics between simulation and Hansen's.

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

# This is used for the root_scalar function
import scipy.optimize as opt

# This is used to calculate the excution time of the while loop
import time

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the nlls() function
from nlls import nlls

# import the hp1() function
from hp1 import hp1


# Model Parameters
alf = 0.36
beta = 0.99
a = 2
delta = 0.025
rho = 0.95
sigmae = 0.00712
Ez = 1   # expectation of shock

# Labor working hour
lmin = 1e-5
lmax = 1-lmin

# Algorithm Parameters
siyes = 1     # simulte model if equals to 1
stationary=1  # finds stationary distribution for capital
tolv = 1e-4   # tolerance level
lam = 0.75    # updating parameter
T = 10000     # simulation period
loadin = 0    # load results


# Define Steady state variables
xx = (1-beta*(1-delta))/(beta*alf*Ez)
yy = ((1/beta+delta-1)/alf*(1+(1-alf)/a)-delta)*a/((1-alf)*Ez)
l_ss = xx/yy  # for this set of parameters, the steady state labor approxiamately equals to 1/3
k_ss = xx**(1/(alf-1))*l_ss
y_ss = Ez*k_ss**alf*l_ss**(1-alf)
i_ss = delta*k_ss
c_ss = y_ss-i_ss



# Simulation of continuous shocks process
zt = np.ones((T+1,1))

rng = np.random.Generator(np.random.MT19937(0))
for t in np.arange(1,T+1):
    zt[t] = np.exp(rho* np.log(zt[t-1]) + sigmae*rng.standard_normal((1,1)))

#zt = zt.T



if loadin == 1:
    with open('parameters.npy', 'rb') as f:
        bet = np.load(f)
else:
    # initial value of parameters
    bet = np.zeros((3,1))
    bet[0] = (alf*k_ss**(alf-1)*l_ss**(1-alf)+1-delta)/c_ss
    bet0 = np.ones((3,1))
    bet = (bet-lam)/(1-lam)

    # Define matrices
    c = np.zeros((T+1,1))
    k = np.zeros((T+1,1))
    y = np.zeros((T+1,1))
    l = np.zeros((T+1,1))
    inv = np.zeros((T+1,1))
    Pea = np.zeros((T+1,1))

    it = 0
    start = time.perf_counter()
    
    while np.abs(bet-bet0).max() > tolv:
        it = it+1
        bet = lam*bet0+(1-lam)*bet
        bet0 = bet.copy()

        # Simulation for given parameters
        k[0] = k_ss
        c[0] = c_ss
        l[0] = l_ss
        inv[0] = i_ss
        y[0] =  y_ss

        for t in np.arange(1,T+1):
            Pea[t]=np.exp(bet[0]+bet[1]*np.log(zt[t])+bet[2]*np.log(k[t-1]))
            c[t]=1/(beta*Pea[t])
            
            func = lambda x: a*c[t]-(1-alf)*zt[t]*k[t-1]**alf*x**(-alf)*(1-x)
            l[t] = opt.root_scalar(func, bracket=[lmin,lmax]).root
            if l[t] < 0:
                l[t] = 0
            elif l[t] > 1:
                l[t] = 1
            
            y[t] = zt[t]*k[t-1]**alf*l[t]**(1-alf)
            k[t] = zt[t]*k[t-1]**alf*l[t]**(1-alf)+(1-delta)*k[t-1]-c[t]
            inv[t] = y[t]-c[t]
        

        E3=((c[2:T+1]**(-1))*(alf*zt[2:T+1]*((k[1:T]**(alf-1))*(l[2:T+1]**(1-alf)))+(1-delta)))

        # calculating the minimizing parameters
        bet = nlls(E3,np.log(zt[1:T]),np.log(k[0:T-1]),T,bet0)
        eva = np.array([np.mean(k), np.mean(c), np.mean(l), np.mean(y), np.mean(inv)])
        print("Means:", eva)
        print("iteration = ", it, ", error = ", np.abs(bet-bet0).max(), sep='')
    
    
    stop = time.perf_counter()
    print("Elapsed time for the while loop is", round(stop - start,4), "seconds")
    
    with open('parameters.npy', 'wb') as f:
        np.save(f, bet)


print(bet)



##############
# Simulation
#############

if siyes:
    T = 115
    N = 100
    ss = np.zeros((N,6))
    cc = np.zeros((N,6))
    
    for j in np.arange(0,N):
        r = rng.standard_normal((T+1,1))
        zt = np.ones((T+1,1))
        zt[0] = 1
        kt = np.zeros((T+1,1))
        kt[0] = k_ss
        it = np.zeros((T,1))
        ct = np.zeros((T,1))
        yt = np.zeros((T,1))
        lt = np.zeros((T,1))
        prodt = np.zeros((T,1))
        Peat = np.zeros((T,1))
        for t in np.arange(0,T):
            zt[t+1] = np.exp(rho* np.log(zt[t]) + sigmae*r[t])
            Peat[t] = np.exp(bet[0]+bet[1]*np.log(zt[t])+bet[2]*np.log(kt[t]))
            ct[t] = 1/(beta*Peat[t])
            func = lambda x: a*ct[t]-(1-alf)*zt[t]*kt[t]**alf*x**(-alf)*(1-x)
            lt[t] = opt.root_scalar(func, bracket=[lmin,lmax]).root
            if lt[t] < 0:
                lt[t] = 0
            elif lt[t] > 1:
                lt[t] = 1
            
            yt[t] = zt[t]*kt[t]**alf*lt[t]**(1-alf)
            kt[t+1] = zt[t]*kt[t]**alf*lt[t]**(1-alf)+(1-delta)*kt[t]-ct[t]
            it[t]= yt[t]-ct[t]
            prodt[t] = yt[t]/lt[t]
            
            
        z = zt[0:T]
        k = np.log(kt[0:T])
        y = np.log(yt[0:T])
        c = np.log(ct)
        i = np.log(it)
        h = np.log(lt[0:T])
        prod = np.log(prodt)
        dhp, dtr = hp1(np.concatenate((y, i, c, k, h, prod),axis=1),1600)
        ss[j,:] = np.std(dhp,axis=0,ddof=1)*100
        Corr = np.corrcoef(dhp,rowvar=False)
        cc[j,:] = Corr[:,0]
        
    std = np.mean(ss,axis=0)
    corr = np.mean(cc,axis=0)
    
    print('HANSEN: std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
    print(np.concatenate((np.array([[1.36, 4.24, 0.42, 0.36, 0.7, 0.68]]).T/1.36, \
                          np.array([[1, 0.99, 0.89, 0.06, 0.98, 0.98]]).T),axis=1))
    print('std(x) std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
    print(np.concatenate((std[:,np.newaxis], (std/std[0])[:,np.newaxis], \
                          corr[:,np.newaxis]),axis=1))
        





