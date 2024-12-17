"""
Simulate discrete finite state Markov chain

Yan Liu, 2011.4.7
"""

# Importing packages
import numpy as np

def mcsim(ss,tm,T):
    # ss is state space, tm is transition matrix, and
    # T is the number of periods to be simulated

    n = ss.size
    
    # stationary distribution
    sd = np.matmul((np.concatenate((np.zeros((1,n - 1))[0],(np.array([1]))))),\
    np.linalg.inv((np.append(tm[0:n - 1,:].T - np.eye(n,n - 1),np.ones((n,1)),1)))) 
    
    sd = sd.T / np.sum(sd)
    
    Ix = np.zeros((T,1),dtype=int)    # indices of the chain
    y = np.random.random((T,1))      # random vector

    Ix[0] = np.argwhere(sd.cumsum() >= y[0])[0]

    tmcum = np.cumsum(tm,1).T    # use the transpose to accelerate computations
    for t in np.arange(1,T):
        Ix[t] = np.argwhere(tmcum[:,Ix[t - 1]] >= y[t])[0,0]

    return Ix



