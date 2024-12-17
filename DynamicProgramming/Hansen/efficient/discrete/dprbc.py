"""
user defined functions for solving the problem

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

# This is used to calculate the excution time of several loops
import time

# utility function
def fu(c, h, a):
    u = (1 - a) * np.log(c) + a * np.log(1 - h) # use Cooley & Prescott (1995) specification 
    return u


# production function
def fprod(k, alpha):
    y = k**alpha
    # In this case it is ok to compute the capital component only
    # since the production function is multiplicative.
    return y

# function determine labor supply given k',k and theta
def fh(h,rik,alpha,a):
    # rik is the ratio of investment and capital (with shock)
    y = ((1 - a) * (1 - alpha) / a + 1) * h**(1 - alpha) -\
        (1 - a) * (1 - alpha) / a * h**(-alpha) - rik
    j = ((1 - a) * (1 - alpha) / a + 1) * (1 - alpha) * h**(-alpha) +\
        (1 - a) * (1 - alpha) * alpha / a * h**(-alpha - 1)
    return y, j


def kronv(x,y):
    # Compute kroneck product of two column vector using basic
    # built-in functions for array manipulation instead of using
    # another built-in function kron, which involves non-basic
    # functions like meshgrid.
    
    # only accepts two-dimensional numpy arrays as inputs
    
    nx = x.size
    ny = y.size

    z = np.reshape(np.matmul(y,x.T),nx * ny,order="F")
    return z

def kronm(x,y):
    # Compute kroneck product of two matrices using kronv
    # only accepts two-dimensional numpy arrays as inputs
    
    nx,kx = x.shape
    ny,ky = y.shape
    
    z = np.zeros((nx * ny,kx * ky))
    for l in np.arange(0,kx):
        for m in np.arange(0,ky):
            z[:,l * ky + m] = kronv(x[:,l][:,np.newaxis],y[:,m][:,np.newaxis])
    
    return z



# It's possible to use built-in function numpy.ravel_multi_index and numpy.unravel_index
def indlin(x,n):
    # Compute the linearized index of an index matrix x, presuming x is
    # generated from an array w which has one more dimension of length n.
    j,k = x.shape
    
    y = x.flatten('F') + kronv(np.ones((k,1)),n * np.arange(0,j)[:,np.newaxis]) + \
        kronv(n * j * np.arange(0,k)[:,np.newaxis],np.ones((j,1)))
    
    return y.astype(int)
              
def indmat(x,n):
    # decompose index matrix x with the range of 1 to n * k into
    # two matrices with ranges of n and k, where y corresponds to n.
    z = np.floor((x - 1)/n)
    y = x - n * z
    
    return y.astype(int), z.astype(int)


# Solve RBC model using dynamic programming for divisible labor supply
import scipy.optimize as opt


def dprbc(me, ir, PAR, DSS, SSK, SSS, GRIDH, TM):
    """ Syntax: [v,gk,gh,gi,gc,gy] = dprbc(me, ir, PAR, DSS, SSK, SSS, GRIDH, TM) 

    ir indicates to irreducible investment,
    with 0 to be reducible and 1 to be irreducible;
    me is the method of solving labor supply
    0 is using FOCs to determine labor given k_t and k_t+1;
    1 is dicretizing labor space
    PAR is parameter vector, DSS is deterministic steady state
    SSS is the state space of shock, TM is the transition matrix
    """
    #--------------------------------------------------------------------------
    if me not in [0,1]:
        print("dprbc input me could only be 0 or 1")
        print("defaulting to me=0")
        me=0
    
    alpha = PAR[0]
    beta = PAR[1]
    delta = PAR[2]
    a = PAR[4]

    cb = DSS[1]
    hb = DSS[2]

    sss = SSS  # column vector
    tm = TM
    ns = sss.size

    ssk = SSK
    nk = ssk.size
    gridh = GRIDH
    nh = gridh.size
    
    # initial value to be half of deterministic steady state
    vnew = 0.5 * fu(cb,hb,a) * np.ones((nk,ns)) / (1 - beta)
    vold = np.zeros((nk,ns))

    # Use focs to determine optimal labor supply for each k.T, k and theta
    iv = kronv(np.ones((nk,1)), ssk[:,np.newaxis]) - kronv((1 - delta) * ssk[:,np.newaxis], np.ones((nk,1)))
    if ir == 1:
        iv = np.maximum(iv,np.zeros((nk**2,1)))
    
    #--------------------------------------------------------------------------
    
    
    if me == 0:
        iv = kronv(np.ones((ns,1)),iv[:,np.newaxis]) # nk^2*ns by 1 array
        
        # for each combination of k.T,k and theta, solve for h
        h = np.zeros((nk**2 * ns,1))   # heuristic state space of h

        prodk = kronv(sss[:,np.newaxis],kronv(fprod(ssk,alpha),np.ones((nk,1)))[:,np.newaxis])
        rik = iv / prodk
        
        start = time.perf_counter()
        for i in np.arange(0,nk**2 * ns):
            fh_i = lambda x: fh(x,rik[i],alpha,a)[0]
            h[i] = opt.fsolve(fh_i,hb/4) # start from 0.1

        stop = time.perf_counter()
        print("Time of solving for h is:", round(stop - start,4), "seconds")
        h = np.minimum(h,1-1e-6)
        h = np.maximum(h,0)   # truncate h at 0 and 1

        # second compute consumption array

        c = prodk*np.squeeze(h)**(1 - alpha)-iv

        Ic = np.argwhere(c >= 0) # indices of infeasible consumption

        # initialize current utility array by assigning large negative value
        u = -0.5 * np.finfo(float).max * np.ones((nk**2 * ns,1))
        
        # utility level is large negative for infeasible consumption
        u[Ic] = fu(c[:,np.newaxis][Ic],h[Ic],a)
        
        u = u.reshape((nk,nk,ns),order="F") # put u into the required array form
        
        
        tol = np.linalg.norm(vnew - vold,1)
        print("norm =", tol)
        nitr = 0
        
        start = time.perf_counter()
        while tol > 1e-6:
            vold = vnew.copy()
            vful = np.matmul(vold, tm.T) # vful is in k.T by k by theta form            

            # Using the built-in numpy.kron is much faster than using the basic kronm and kronv 
            #vfuld = np.reshape(kronm(np.ones((nk,1)),vful),(nk,nk,ns), order='F')
            vful = np.reshape(np.kron(np.ones((nk,1)), vful), (nk,nk,ns), order='F') 

            vnew = np.amax(u + beta * vful, axis=0)
            
            tol = np.linalg.norm(vnew - vold,1)
            print("norm =", tol)
            nitr = nitr + 1
            

        stop = time.perf_counter()
        print("Time of while loop of computing h using FOC is:", round(stop - start,4), "seconds")   
        print('Number of iterations of the while loop:', nitr)
        
        vful = np.matmul(vnew, tm.T)
        vful = np.reshape(kronm(np.ones((nk,1)), vful), (nk,nk,ns), order='F')
        v, Igk = (u + beta * vful).max(axis=0), (u + beta * vful).argmax(axis=0) # Igk contains indices of gk
        
        gk = ssk.squeeze()[Igk]
        Iopt = np.reshape(indlin(Igk,nk),(nk,ns),order='F')

        # compute the linearized index for the optimal choice array of the
        # form k.T by k by theta
        gi = iv[Iopt]
        gh = h.squeeze()[Iopt]
        gc = c[Iopt]
        gy = gc + gi
        
        
    elif me == 1:
        # Use discret h.

        iv = kronv(kronv(np.ones((ns,1)),iv[:,np.newaxis])[:,np.newaxis],np.ones((nh,1))) # nh*nk^2*ns by 1 array
        
        c = kronv(sss[:,np.newaxis],kronv(fprod(ssk[:,np.newaxis],alpha),\
                                          kronv(np.ones((nk,1)),gridh**(1-alpha))[:,np.newaxis])[:,np.newaxis]) - iv
        
        Ic = np.argwhere(c >= 0) # indices of infeasible consumption
        
       
        h = kronv(np.ones((ns,1)),kronv(np.ones((nk,1)),kronv((np.ones((nk,1))),gridh)[:,np.newaxis])[:,np.newaxis])
        
        
        # initialize current utility array by assigning large negative value
        u = -0.5 * np.finfo(float).max * np.ones((nh * nk**2 * ns,1))
        # utility level is large negative for infeasible consumption
        u[Ic] = fu(c[:,np.newaxis][Ic],h[:,np.newaxis][Ic],a)
        u = u.reshape((nk*nh,nk,ns),order="F") # put u into the required array form
        
        
        tol = np.linalg.norm(vnew - vold,1)
        print("norm =", tol)
        nitr = 0
        
        start = time.perf_counter()
        while tol > 1e-6:
            vold = vnew.copy()
            vful = np.matmul(vold, tm.T) # vful is in k.T by k by theta form            

            # Using the built-in numpy.kron is much faster than using the basic kronm and kronv 
            #vfuld = np.reshape(kronm(np.ones((nk,1)),vful),(nk,nk,ns), order='F')
            vful = np.reshape(np.kron(np.ones((nk,1)), np.kron(vful,np.ones((nh,1)))), (nk*nh,nk,ns), order='F') 

            vnew = np.amax(u + beta * vful, axis=0)
            
            tol = np.linalg.norm(vnew - vold,1)
            print("norm =", tol)
            nitr = nitr + 1
            

        stop = time.perf_counter()
        print("Time of while loop of computing h using FOC is:", round(stop - start,4), "seconds")   
        print('Number of iterations of the while loop:', nitr)
        
        vful = np.matmul(vnew, tm.T)
        vful = np.reshape(kronm(np.ones((nk,1)), kronm(vful,np.ones((nh,1)))), (nk*nh,nk,ns), order='F') 
        
        
        v, Ighk = (u + beta * vful).max(axis=0), (u + beta * vful).argmax(axis=0) # Igk contains indices of gh and gk
        
        Igh,Igk = indmat(Ighk,nh)
        
        gk = ssk.squeeze()[Igk]
        gh = gridh.squeeze()[Igh]
        gi = gk - (1 - delta) * np.matmul(ssk, np.ones((1,ns)))
        gy = np.matmul(fprod(ssk,alpha), sss[np.newaxis,:])*gh**(1 - alpha)
        gc = gy - gi        
        
    
    return v,gk,gh,gi,gc,gy
