"""

This file contains the function maxlininterpol()

Translated from Eva Carceles-Poveda's MATLAB codes
"""

import numpy as np

def maxlininterpol(kpr, V0, beta, inc, Pi, j, k):
    pos=np.abs(kpr-k).argmin()

    if k[pos] > kpr:
        weight = (k[pos]-kpr)/(k[pos]-k[pos-1])
        y = np.log(inc-kpr) + beta*np.dot(weight*V0[pos-1,:]+(1-weight)*V0[pos,:],Pi[j,:]).T
    elif k[pos] == kpr:
        y = np.log(inc-kpr) + beta*np.dot(V0[pos,:],Pi[j,:]).T
    else:
        weight = (kpr-k[pos])/(k[pos+1]-k[pos])
        y = np.log(inc-kpr) + beta*np.dot(weight*V0[pos+1,:]+(1-weight)*V0[pos,:],Pi[j,:]).T

    y=-y;
    #print('y=',y)
    return y