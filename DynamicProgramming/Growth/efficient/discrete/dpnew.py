"""
Contains the dpnew, fu, and frpod functions

Translated from Eva Carceles-Poveda's MATLAB codes
"""

import numpy as np

# This is used to calculate the excution time of the while loop
import time

# utility function
def fu(x, gamma):
    if gamma != 1:
        y = x**(1 - gamma)/(1 - gamma)
    else:
        y = np.log(x)
    
    return y


# production function
def fprod(x, alpha):
    y = x**alpha
    return y


def dpnew(ir, Kgrid, PAR, DSS, SSS, TM):
    """ Syntax: [vnew, gk] = dpnew(ir, Kgrid, PAR, DSS, SSS, TM)

    ir is 0 or 1, where 0 refers to the default option of reversible investment
    kgrid is the capital grid
    PAR is the parameter vector, DSS is the deterministic steady state
    SSS is the state space of shock, TM is the transition matrix
    """

    beta = PAR[1]
    delta = PAR[3]

    cb = DSS[1]

    sss = SSS
    tm = TM
    ns = sss.size

    kg = Kgrid
    nk = kg.size

    # initial value to be half of deterministic steady state
    vold = np.zeros((nk,ns))
    vnew = 0.5 * fu(cb,PAR[2]) * np.ones((nk,ns)) / (1 - beta)

    gk = np.zeros((nk,ns))
    
    
    #required array of consumption combination, c(k',k,theta), in an
    #nk^2 * ns by 1 form

    # first create an array of investment
    iv = np.kron(np.ones((nk,)), kg) - np.kron((1 - delta) * kg, np.ones((nk,)))
    if ir == 1:
        iv = np.maximum(iv,np.zeros((nk**2,)))
        
    iv = np.kron(np.ones((ns,)),iv)   #nk^2*ns by 1 array
    
    # compute consumption array
    c = np.kron(sss,np.kron(fprod(kg,PAR[0]),np.ones((nk,)))) - iv
    Ic = np.argwhere(c >= 0)  # indices of infeasible consumption
    u = -0.5 * np.finfo(float).max * np.ones((nk**2 * ns,))  # initialize current utility array
    u[Ic] = fu(c[Ic],PAR[2])  # utility level is -inf for infeasible consumption
    u = u.reshape((nk,nk,ns),order="F")  # put u into the required array form

    print("norm = ", np.linalg.norm(vold-vnew,1))
    
    start = time.perf_counter()
    while np.linalg.norm(vold - vnew,1) > 1e-6:
        vold = vnew.copy()
        vful = np.dot(vold, tm)  # vful is in k' by k by theta form
        vful = np.reshape(np.kron(np.ones((nk,)),vful),(nk,nk,ns))
        vnew, gk = (u + beta * vful).max(axis=0), (u + beta * vful).argmax(axis=0)
        print("norm = ", np.linalg.norm(vold-vnew,1))

    stop = time.perf_counter()
    print("Elapsed time in seconds for the while loop is:", round(stop - start,4))
    
    return vnew, gk