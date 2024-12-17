# Importing packages
import numpy as np
from scipy.optimize import root_scalar

def lfoc(x,zt,ki,kpr,alf,a,delta):
    c = (1-alf) * zt*ki**alf*x**(-alf)*(1-x)/a
    F = c+kpr-(1-delta)*ki-zt*ki**alf*x**(1-alf)
    return F

def maxinter(kpr, alf, k, beta, zt, t, i, a, delta, lmin, lmax, V0, Tran):

    pos = np.abs(kpr-k).argmin()
    
    global l, c
    
    l = root_scalar(lfoc, args=(zt,k[i],kpr,alf,a,delta), bracket=[lmin,lmax]).root

    if l < 0:
        l = lmin

    if l > 1:
        l = lmax

    c = (1-alf)*zt*k[i]**alf*l**(-alf)*(1-l)/a

    if k[pos] > kpr:
        weight = (k[pos]-kpr)/(k[pos]-k[pos-1])
        y = np.log(c) + a*np.log(1-l) + beta*np.matmul(weight*V0[pos-1,:] + \
                                                       (1-weight)*V0[pos,:],Tran[t,:])
    elif k[pos] == kpr:
        y = np.log(c) + a*np.log(1-l) + beta*np.matmul(V0[pos,:],Tran[t,:])
    else:
        weight = (kpr-k[pos])/(k[pos+1]-k[pos])
        y = np.log(c) + a*np.log(1-l) + beta*np.matmul(weight*V0[pos+1,:]+\
                                                       (1-weight)*V0[pos,:],Tran[t,:])
        
    y = -y

    return y