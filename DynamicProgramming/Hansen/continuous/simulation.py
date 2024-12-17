# Importing packages
import numpy as np
from scipy.optimize import root_scalar


def llfoc(x,zt,ktime,kpr,alf,a,delta):
    c = (1-alf) * zt*ktime**alf*x**(-alf)*(1-x)/a
    F = c+kpr-(1-delta)*ktime-zt*ktime**alf*x**(1-alf)
    return F


def simulation(kpr, k, beta, time, m, a, V0, Tran, tind, zt, alf, ktime, lmin, lmax, delta):

    pos = np.abs(kpr-k).argmin()
    
    global l, c
    
    l = root_scalar(llfoc, args=(zt,ktime,kpr,alf,a,delta), bracket=[lmin,lmax]).root

    if l < 0:
        l = lmin

    if l > 1:
        l = lmax

    c = (1-alf)*zt*ktime**alf*l**(-alf)*(1-l)/a

    if k[pos] > kpr:
        weight = (k[pos]-kpr)/(k[pos]-k[pos-1])
        y = np.log(c) + a*np.log(1-l) + beta*np.matmul(weight*V0[pos-1,:] + \
                                                       (1-weight)*V0[pos,:],Tran[tind[m,time],:])
    elif k[pos] == kpr:
        y = np.log(c) + a*np.log(1-l) + beta*np.matmul(V0[pos,:],Tran[tind[m,time],:])
    else:
        weight = (kpr-k[pos])/(k[pos+1]-k[pos])
        y = np.log(c) + a*np.log(1-l) + beta*np.matmul(weight*V0[pos+1,:]+\
                                                       (1-weight)*V0[pos,:],Tran[tind[m,time],:])
        
    y = -y

    return y
