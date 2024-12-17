# -*- coding: utf-8 -*-
"""

This file contains all the fucntions in the Markov Shocks module. 
To use any of them, you can just copy the function you need 
into your own file, or import them from this file

Translated from Eva Carceles-Poveda's MATLAB codes
"""

import numpy as np # needed for all the functions
from scipy.stats import norm # needed for the markov_approx function only

# lnshock: generates a lognormal shock
def ln_shock(rho, sigmae, T = 200):
    """ Syntax: y=ln_shock(rho,sigmae,T)
    
    This file generates a T-dimensional lognormal shock with mean zero, 
    persistence rho and innovation standard deviation sigmae.
    """
    r = np.random.standard_normal((T+1,1))
    y = np.ones((T+1,1))
    for i in np.arange(2,T+1):
         y[i]=np.exp(rho * np.log(y[i-1]) + sigmae * r[i])
    return y

# lnshockm: generates a lognormal shock
def ln_shock_m(mu, rho, sigmae, T = 200):
    """ Syntax: y=ln_shock_m(mu,rho,sigmae,T)
    
    This file generates a T-dimensional lognormal shock with mean mu,
    persistence rho and innovation standard deviation sigmae.
    """
    
    r = np.random.standard_normal((T+1,1))
    y = np.ones((T+1,1)) * np.exp(mu/(1-rho))
    for i in np.arange(2,T+1):
         y[i]=np.exp(mu + rho * np.log(y[i-1]) + sigmae * r[i])
    return y


# markovchain: generates a Markov chain
def markov_chain(Trans, T = 100, s0 = 0, me = 1):
    """ Syntax: y=markov_chain(Trans,T,s0,me)
    
    This function generates a simulation from a Markov chain.
    Trans is the transition matrix, T is the number of periods 
    to be simulated, s0 is the initial state, with one as default, 
    me is the method (two possibilities are included, with the first 
    as default), y is the shock realization.
    """
    
    Trans = Trans.astype(float) # convert transition matrix to float for precision
    
    # Checking the mistakes from the inputs
    
    num_rows, num_cols = Trans.shape
    
    # check that Trans and s0 are well defined
    if num_rows != num_cols:
        print('Transition matrix must be square')
        return "markov_chain function did not complete execution. Please fix and run again."
    # return
    
    for k in np.arange(0,num_rows):
        if not np.isclose(Trans[k,:].sum(), 1):
            print('Row', k, ' does not sum to one')
            print('Normalizing row', k)
            Trans[k,:] = Trans[k,:] / Trans[k,:].sum()
    
    print("Transition Matrix:\n", Trans)
    
    if s0 not in np.arange(0, num_rows):
        print('Initial state', s0, ' is out of range')
        print('Initial state defaulting to 0')
        s0=0
        
    if me not in np.arange(1,3):
        print("There are only method 1 and method 2")
        print("Method defaulting to method 1")
        me = 1

    #rng = np.random.default_rng(3108)
    #X = rng.random((T,1))
    # Creating the shock realizations
    X = np.random.random((T,1))
    y = np.full((T,1),s0)
    #y[0,0] = s0
    
    if me == 1:
        for i in np.arange(1, T):
            for j in np.arange(0, num_rows):
                if X[i-1] < Trans[y[i-1],0:j+1].sum():
                    break
                j = j + 1
            y[i,0] = j 
        y = y.T   
    elif me == 2:
        s = np.zeros((num_rows,1))
        s[s0] = 1
        cum = np.matmul(Trans, np.triu(np.ones(Trans.shape)))
        state = np.zeros((num_rows, T))
        for k in np.arange(0,T):
            state[:,k] = s[:,0]
            ppi = np.concatenate((np.zeros((1,1)), np.matmul(s.T,cum)), axis = 1)
            s=((X[k]<=ppi[:, 1:num_rows+1])*(X[k]>ppi[:, 0:num_rows])).T

        y = np.matmul(np.arange(0, num_rows),state)   
        y = np.reshape(y,(T,1)).T.astype('int')
    
    
    return y


# ergodic: calculates the stationary distribution of a Markov chain
def ergodic(Trans, me = 1):
    """ Syntax: y=ergodic(Trans, me)
    
    This function calculates the stationary distribution of a 
    Markov chain. Trans is the transition matrix and me is the 
    method (three possibilities are included, with the first 
    as default), P is the distribution.
    """
    
    Trans = Trans.astype(float) # convert transition matrix to float for precision
    
    # Checking the mistakes from the inputs
    
    num_rows, num_cols = Trans.shape
    
    # check that Trans and s0 are well defined
    if num_rows != num_cols:
        print('Transition matrix must be square')
        return "ergodic function did not complete execution. Please fix and run again."
    # return
    
    for k in np.arange(0,num_rows):
        if not np.isclose(Trans[k,:].sum(), 1):
            print('Row', k, ' does not sum to one')
            print('Normalizing row', k)
            Trans[k,:] = Trans[k,:] / Trans[k,:].sum()
    
    print("Transition Matrix:\n", Trans)
        
    if me not in np.arange(1,4):
        print("There are only methods 1, 2, or 3")
        print("Method defaulting to method 1")
        me = 1

        
        
    va, vec = np.linalg.eig(Trans.T)
    va = np.diag(va)
    r1, r2 = (np.absolute(va-1)<1e-14).nonzero()
    
    if r1.size == 1:
        if me == 1:
            M = np.identity(num_rows) - Trans.T
            one = np.ones((1,num_rows))
            MM = np.concatenate((M[0:num_rows-1,:], one), axis = 0)
            V = np.concatenate((np.zeros((num_rows-1,1)), np.ones((1,1))), axis = 0)
            P = np.matmul(np.linalg.inv(MM), V)
            
        elif me == 2:
            trans = Trans.T
            p0 = (1/num_rows)*np.ones((num_rows,1))
            test = 1
            while test > 1e-5:
                p1 = np.matmul(trans, p0)
                test = np.absolute(p1-p0).max()
                p0 = p1
            P=p0
            
        elif me == 3:
            P = np.linalg.matrix_power(Trans,1000000)
            P = P[1,:].T
            P = P[:, np.newaxis] # view P as a column vector
            
    else:
        print('Sorry, there is more than one distribution. All of them, subject to normalization if not')
        print('summing up to one, are the columns of the matrix corresponding to the unity eigenvalues in va') 
        P = vec
        print("va =", va)
    
    
    return P



# iid: generates an iid shock
def iid(prob, T = 100, me = 1):
    """ Syntax: y=iid(prob, T, me)
    
    This function generates a simulation from an iid random variable 
    prob is the probability vector, T is the number of periods to be 
    simulated (default is 100), me is the method (two possibilities 
    are included and it is the first by default), y is the shock realization.
    """
    
    if me not in np.arange(1,3):
        print("There are only method 1 and method 2")
        print("Method defaulting to method 1")
        me = 1
    
    # Checking the mistakes from the inputs
    if not np.isscalar(T):
        print("The number of realizations must be a scalar")
        return "iid function did not complete execution. Please fix and run again."
        
        
    # if the number of realizations in not an natural number, 
    # the following takes the abs. value and rounds to the nearest natural number.    
    T = np.around(np.absolute(T)) 

    # checking if prob is a vector or not
    if np.squeeze(prob).ndim != 1:
        print("prob must be a vector of probabilities")
        return "iid function did not complete execution. Please fix and run again."
    
    prob = np.absolute(prob)
    if not np.isclose(prob.sum(), 1):
        print('The probabilities do not sum to one. Will normalize probabilities.')
        prob = prob / prob.sum()
    
    print("Probability Matrix:\n", prob)
    
    X = np.random.random((T,1))
    m = prob.size
    y = np.zeros((T,1))

    if me == 1:
        for i in np.arange(0, T):
            for j in np.arange(0, m):
                if X[i] < prob[0:j].sum():
                    break
                j = j + 1
            y[i,0] = j 
        
    elif me == 2:
        P = prob.cumsum(axis = 0)  # creates a vector of cumulated probabilities
        for i in np.arange(0, T):
            j, k = (X[i] < P).nonzero()
            y[i,0] = j[0]+1
    
    
    return y



# markovapprox: approximates a continuous AR(1) process with a Markov chain
# Eva Carceles-Poveda's version. 
# You can use this function, Floden's tauchen function, 
# or Sargent's tauchen function.
def markov_approx(rho, sigma, m, N):
    """ Syntax: [Tran,s,p,arho,asigma]=markovapprox(rho,sigma,m,N)
    
    This function approximates a first-order autoregressive process 
    with persistence rho and innovation standard deviation sigma with 
    an N state Markov chain; m determines the width of discretized state 
    space, Tauchen uses m=3, with ymax=m*vary,ymin=-m*vary, where ymax 
    and ymin are the two boundary points, Tran is the transition matrix 
    of the Markov chain, s is the discretized state space, p is the 
    chain stationary distribution, arho is the theoretical first order 
    autoregression coefficient for the Markov chain, asigma is the 
    theoretical standard deviation for the Markov chain.
    
    Translated from Eva Carceles-Poveda 2003 MATLAB code
    """
    
    # Discretize the state space
    stvy = np.sqrt(sigma**2/(1-rho**2))   # standard deviation of y(t)
    ymax = m*stvy                         # upper boundary of state space
    ymin = -ymax                          # lower boundary of state space
    w = (ymax-ymin)/(N-1)                 # distance between points
    s = w * np.arange(ymin/w, ymax/w+1)   # the discretized state space        
    
    
    # Calculate the transition matrix
    Tran = np.zeros((N,N))
    for j in np.arange(0,N):
        for k in np.arange(1,N-1):
            Tran[j,k] = norm.cdf(s[k]-rho*s[j]+w/2,0,sigma) - norm.cdf(s[k]-rho*s[j]-w/2,0,sigma);
            
        Tran[j,0] = norm.cdf(s[0]-rho*s[j]+w/2,0,sigma);
        Tran[j,N-1] = 1 - norm.cdf(s[N-1]-rho*s[j]-w/2,0,sigma);
        
    # Check that Tran is well specified
    if not np.all(np.isclose(np.sum(Tran.T, axis=0), np.squeeze(np.ones((1,N))))):
        # find rows not adding up to one
        str = (np.absolute(np.sum(Tran.T, axis=0))-np.squeeze(np.ones((1,N)))<1e-14).nonzero()          
        print('error in transition matrix')
        print('rows', str[0],' do not sum to one')
    
    
    # Calculate the invariant distribution of Markov chain
    Trans = Tran.T
    p = (1/N)*np.ones((N,1)) # initial distribution of states
    test = 1;
    while test > 1e-8:
        p1 = np.matmul(Trans,p)
        test=np.max(np.abs(p1-p))
        p = p1
    
    
    meanm = np.matmul(s,p)            # mean of invariant distribution of chain
    varm = np.matmul((s-meanm)**2,p)  #variance of invariant distribution of chain  
    midaut1 = np.matmul((s-meanm)[:, np.newaxis],(s-meanm)[np.newaxis, :]) # cross product of deviation from mean of yt and yt-1                    
    probmat = np.matmul(p,np.ones((1,N)))     # each column is invariant distribution   
    midaut2 = Tran*probmat*midaut1 # product of the first two terms is joint distribution of (Yt-1,Yt)                                    
    autcov1 = np.sum(midaut2)    #  first-order auto-covariance
    
    arho = autcov1/varm           # theoretical first order autoregression coefficient
    asigma = np.sqrt(varm)           # theoretical standard deviation
    
    return Tran, s, p, arho, asigma