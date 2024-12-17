"""
Hansen with DDP

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

import matplotlib.pyplot as plt

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the acm() function
from acm import acm

# import the hp1() function
from hp1 import hp1

# import the dprbc() and mcsim() functions
from dprbc import dprbc
from mcsim import mcsim

FO = 1

alpha = 0.36
beta = 0.99
gamma = 1       # log utility
delta = 0.025
a = 2/3         # correspond to A = a / (1 - a) = 2 in Hansen (1985)
PAR = np.array([alpha,beta,delta,gamma,a])  # a governs labor supply

SSS, TM = acm(0,0.95,0.00712,7,1)[0:2]
SSS = np.exp(SSS)     # acm returns the log productivity shock

# rkh is the ratio of kb and hb
rkh = (1 / alpha / beta + (delta - 1) / alpha)**(1 / (1 - alpha))

x = np.linalg.solve(np.array([[rkh, a * rkh**alpha / (1 - a) /\
                               (1 - alpha)],[rkh**(1 - alpha) - delta, -1]]),\
                   np.array([[1],[0]]))

kb = x[0]
cb = x[1]
ib = delta * kb
hb = rkh * kb
yb = cb + ib
DSS = np.array([kb,cb,hb,ib,yb])


range_k = np.array([0.894,1.115]) # around steady state
nk = 200
SSK = np.linspace(range_k[0] * kb, range_k[1] * kb, nk)


if FO == 1:
    # Hansen with Discrete Dynamic Programming: FOC for h
    nh = 100
    range_h = (range_k - 1) * 2/3 + 1
    GRIDH = np.linspace(range_h[0] * hb, min(range_h[1] * hb,1), nh)
    
    v, gk, gh, gi, gc, gy = dprbc(0,0,PAR, DSS, SSK, SSS, GRIDH, TM)
elif FO == 0:
    # Hansen with Discrete Dynamic Programming: Discretization of h
    nh = 50
    range_h = (range_k - 1) * 2/3 + 1
    GRIDH = np.linspace(range_h[0] * hb, min(range_h[1] * hb,1), nh)
    
    v, gk, gh, gi, gc, gy = dprbc(1,0,PAR, DSS, SSK, SSS, GRIDH, TM)
    


fig1, axs1 = plt.subplots(3, 2)

axs1[0,0].plot(SSK, gk[:,0], 'k-')
axs1[0,0].plot(SSK, gk[:,-1], 'k--')
axs1[0,0].set(xlabel='$k$', title='$g_{k}(k)$')

axs1[0,1].plot(SSK, gh[:,0], 'k-')
axs1[0,1].plot(SSK, gh[:,-1], 'k--')
axs1[0,1].set(xlabel='$k$', title='$g_{h}(k)$')

axs1[1,0].plot(SSK, gi[:,0], 'k-')
axs1[1,0].plot(SSK, gi[:,-1], 'k--')
axs1[1,0].set(xlabel='$k$', title='$g_{i}(k)$')

axs1[1,1].plot(SSK, gc[:,0], 'k-')
axs1[1,1].plot(SSK, gc[:,-1], 'k--')
axs1[1,1].set(xlabel='$k$', title='$g_{c}(k)$')

axs1[2,0].plot(SSK, gy[:,0], 'k-')
axs1[2,0].plot(SSK, gy[:,-1], 'k--')
axs1[2,0].set(xlabel='$k$', title='$g_{y}(k)$')

axs1[2,1].plot(SSK, v[:,0], 'k-')
axs1[2,1].plot(SSK, v[:,-1], 'k--')
axs1[2,1].set(xlabel='$k$', title='$V(k)$')

plt.tight_layout()
plt.savefig('Hansen_DDP_eff_policy_value.jpg', dpi=800)
plt.show()
plt.close(fig1)


#--------------------------------------------------------------------------
# Simulation for Hansen model with divisible labor
#--------------------------------------------------------------------------
#np.random.seed(1337)

ss = SSS
tm = TM
ns = ss.size

Igk = np.around((gk - SSK[0]) / (SSK[-1] - SSK[0]) * (nk - 1))-1

T = 115
N = 100

Ik = np.zeros((T,1),dtype=int)
Ik[0] = np.floor(SSK.size/2)-1   # Always start from kb (nearby).

ss_mat = np.zeros((N,6))
cc_mat = np.zeros((N,6))

for j in np.arange(0,N):
    Is = mcsim(ss,tm,T)
    
    for t in np.arange(1,T):
        Ik[t] = Igk[Ik[t-1],Is[t-1]]
    
    k = np.log(SSK[Ik.squeeze()])
    inv = np.log(gi[Ik,Is])
    c = np.log(gc[Ik,Is])
    h = np.log(gh[Ik,Is])
    y = np.log(gy[Ik,Is])
    prod = y - h  # productivity
    
    dhp, dtr = hp1(np.concatenate((y, inv, c, k, h, prod),axis=1), 1600)
    ss_mat[j,:] = np.std(dhp,axis=0,ddof=1)*100
    Corr = np.corrcoef(dhp,rowvar=False)
    cc_mat[j,:] = Corr[:,0]
    

std = np.mean(ss_mat,axis=0)
corr = np.mean(cc_mat,axis=0)

print('HANSEN: std(x)/std(y) corr(x,y) for y, i, c, k, h, prod')
print(np.concatenate((np.array([[1.36, 4.24, 0.42, 0.36, 0.7, 0.68]]).T/1.36, \
                  np.array([[1, 0.99, 0.89, 0.06, 0.98, 0.98]]).T),axis=1))
print('std(x) std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
print(np.concatenate((std[:,np.newaxis], (std/std[0])[:,np.newaxis], \
                      corr[:,np.newaxis]),axis=1))





