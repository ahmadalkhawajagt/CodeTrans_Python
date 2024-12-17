# -*- coding: utf-8 -*-
"""
This file contains the functions from Flodén. 
To use any of them, just copy the function you need 
into your own file, or import them from this file

Translated from Flodén's MATLAB codes
"""
# needed for both addacooper and tauchen
import numpy as np
from scipy.stats import norm

# needed for addacooper only
from scipy.integrate import quad 

# generates an AR(1) process using Adda and Cooper's (Dynamic Economics, 2003) method
def addacooper(n,mu,rho,sigma):
    """ Syntax: [Z,PI] = addacooper(n,mu,rho,sigma);

    Approximate n-state AR(1) process following Tauchen (1986) and Tauchen & Hussey (1991).
    See Adda & Cooper (2003) pp 57-.
    Z(t+1) = mu*(1-rho) + rho*Z(t) + eps(t+1)
    where std(eps) = sigma
    Translated from Martin Flodén, 2005
    """

    sigmaUNC = sigma/np.sqrt(1-rho**2)
    E  = np.zeros((n+1,1))
    Z  = np.zeros((n,1))
    PI = np.zeros((n,n))
    MFPI = np.zeros((n,n))
    
    E[0] = -1e1
    E[-1] = 1e1
    for i in np.arange(1,n):
        E[i] = sigmaUNC*norm.ppf((i)/n) + mu
    
    for i in np.arange(0,n):
        Z[i] = n*sigmaUNC*(norm.pdf((E[i]-mu)/sigmaUNC) - norm.pdf((E[i+1]-mu)/sigmaUNC)) + mu
    

    for i in np.arange(0,n):
        for j in np.arange(0,n):
            E1 = E[j]
            E2 = E[j+1]
            th_fcn = lambda u: n/np.sqrt(2*np.pi*sigmaUNC**2) * (np.exp(-(u-mu)**2 / (2*sigmaUNC**2)) * \
                      (norm.cdf((E2-mu*(1-rho)-rho*u)/sigma) - norm.cdf((E1-mu*(1-rho)-rho*u)/sigma)))
            
            PI[i,j] = quad(th_fcn,E[i],E[i+1],epsabs=1e-10)[0]
            MFPI[i,j] = norm.cdf((E[j+1]-mu*(1-rho)-rho*Z[i])/sigma) - norm.cdf((E[j]-mu*(1-rho)-rho*Z[i])/sigma)       


    for i in np.arange(0,n):
        PI[i,:] = PI[i,:] / np.sum(PI[i,:])
        MFPI[i,:] = MFPI[i,:] / np.sum(MFPI[i,:])

    return Z, PI


# generates an AR(1) process using Tauchen's (Ec. Letters, 1986) method
def tauchen(N,mu,rho,sigma,m):
    """ Syntax: [Z,Zprob] = tauchen(N,mu,rho,sigma,m)
    
    Function TAUCHEN
    Purpose:    Finds a Markov chain whose sample paths
                approximate those of the AR(1) process
                    z(t+1) = (1-rho)*mu + rho * z(t) + eps(t+1)
                where eps are normal with stddev sigma
                
    Format:     {Z, Zprob} = Tauchen(N,mu,rho,sigma,m)
    
    Input:      N       scalar, number of nodes for Z
                mu      scalar, unconditional mean of process
                rho     scalar
                sigma   scalar, std. dev. of epsilons
                m       max +- std. devs.

    Output:     Z       N*1 vector, nodes for Z
                Zprob   N*N matrix, transition probabilities

    This procedure is an implementation of George Tauchen's algorithm
    described in Ec. Letters 20 (1986) 177-181.

    Translated from Martin Flodén, Fall 1996
    """

    
    
    
    Z     = np.zeros((N,1))
    Zprob = np.zeros((N,N))
    a     = (1-rho)*mu

    Z[-1]  = m * np.sqrt(sigma**2 / (1 - rho**2))
    Z[0]  = -Z[-1]
    zstep = (Z[-1] - Z[0]) / (N - 1)
    
    
    for i in np.arange(1,N-1):
        Z[i] = Z[0] + zstep * i


    Z = Z + a / (1-rho)
    

    for j in np.arange(0,N):
        for k in np.arange(0,N):
            if k == 0:
                Zprob[j,k] = norm.cdf((Z[0] - a - rho * Z[j] + zstep / 2) / sigma)
            elif k == N-1:
                Zprob[j,k] = 1 - norm.cdf((Z[-1] - a - rho * Z[j] - zstep / 2) / sigma)
            else:
                Zprob[j,k] = norm.cdf((Z[k] - a - rho * Z[j] + zstep / 2) / sigma) - \
                         norm.cdf((Z[k] - a - rho * Z[j] - zstep / 2) / sigma)
    
    return Z, Zprob