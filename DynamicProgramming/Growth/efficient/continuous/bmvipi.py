"""
user defined functions for solving the problem

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

# This is used to calculate the excution time of several loops
import time

# This is used for the root, fsolve, and fminbound function
import scipy.optimize as opt

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


def linitp(x,x_grid,y_grid):
    # x_grid is a vector summarizing the information of the grid of x,
    # including the lower bound x_grid(1) and step size x_grid(2).
    # y_grid has the same number of columns as x, while the length of x
    # could be greater than y_grid. Users are required to make
    # sure x lies in the admissible region.

    k = x.shape[0]
    n,m = y_grid.shape

    Ix = np.fix((x - x_grid[0]) / x_grid[1])
    w = (x - x_grid[0]) / x_grid[1] - Ix

    arr_index1 = (np.maximum(Ix,0).astype(int),\
                 np.matmul(np.ones((k,1)), np.arange(0,m)[np.newaxis,:]).astype(int))
    #Ind1 = np.ravel_multi_index(arr_index1, (n,m), order='F') # equivalent to sub2ind in Matlab
    arr_index2 = (np.minimum(Ix+1,n-1).astype(int),\
                  np.matmul(np.ones((k,1)), np.arange(0,m)[np.newaxis,:]).astype(int))
    #Ind2 = np.ravel_multi_index(arr_index2, (n,m), order='F') # equivalent to sub2ind in Matlab

    y = (1 - w) * y_grid[arr_index1] + w * y_grid[arr_index2]

    return y


# utility function
def fu(x, gamma):
    if gamma == 1:
        y = np.log(x)
    else:
        y = x**(1 - gamma)/(1 - gamma)
        
    return y


# production function
def fprod(x, alpha):
    y = x**alpha
    return y


def fv(gc,gk,tmt,ssk_grid,beta,gamma):
    # solve for value function using gc, gk and tmt
    
    u = fu(gc,gamma)  # current utility
    
    # functional equation for v
    # scipy.optimize victorizes the input and output of the function, 
    # so we must reshape the input v to its original shape, 
    # and we must flatten the output of the function
    fneq = lambda v: (u + beta * np.matmul(linitp(gk,ssk_grid,v.reshape(gk.shape)),tmt) \
                      - v.reshape(gk.shape)).flatten()
    
    v0 = 1 * u / (1 - beta)

    v = opt.root(fneq,v0,method="lm").x.reshape(gk.shape)
    # or we can use the following:
    #v = opt.fsolve(fneq,v0,method="lm").reshape(gk.shape)
    return v



# Uses value function interpolation or policy function iteration to solve
# Brock-Mirman optimal growth model.
# Translated from Yan Liu's Matlab code, 2011.4.9
def bmvipi(me,vime,PAR, DSS, SSK, SSS, TM):
    """ Syntax: [v,gk,gc,gi,gy,t] = bmvipi(me,vime,PAR DSS SSK SSS TM) 

    me = 1 corresponds to value function interpolation and me = 0
    corresponds to policy function iteration. vime indicates the method used
    in value function interpolation, with vime = 0 referring to discrete
    maximization and vime = 1 referring to sophiscated optimization (fminbnd alike).
    """
    #--------------------------------------------------------------------------

    # vful_j is used to define maxv in value function iterpolation with continuous optimization
    alpha = PAR[0]
    beta = PAR[1]
    delta = PAR[2]
    gamma = PAR[3]
    bk = DSS[0]  # use the syntax b* to denote steady state value
    bc = DSS[1]
    ssk = SSK  # nkx1
    nk = ssk.size
    sss = SSS.T     # ns x1
    ns = sss.size
    tmt = TM
    
    t = {'pi':0,'vi_dsc':0,'vi_ctn':0}
    ssk_grid = np.array([ssk[0],(ssk[-1] - ssk[0]) / (nk - 1)])
    # ssk_grid is used by all methods.
    
    #--------------------------------------------------------------------------
    
    
    if me == 0:
        # use policy iteration with linear iterpolation
        tm1 = kronm(tmt,np.append(1,np.zeros((1,ns)))[np.newaxis,:])
        tm1 = np.reshape(tm1[:,0:ns**2],(ns**2,ns),order="F")    # for computing E_t...
        
        gk_new = (1 - (1 - alpha) * delta) * np.matmul((ssk - bk)[:,np.newaxis], \
                                                       np.ones((1,ns))) + \
            bk * np.matmul(np.ones((nk,1)), (1 * (sss[np.newaxis,:] - 1) + 1))
        #gk_new = (1 - (1 - alpha) * delta) * np.matmul((ssk - bk)[:,np.newaxis], \
        #                                               np.ones((1,ns))) + \
        #    bk * np.ones((nk,ns))
                
        gk_new = np.maximum(gk_new,ssk[0])
        gk_new = np.minimum(gk_new,ssk[-1])
        # This choice is motivated by the approximation around steady state.
        
        gk_old = np.zeros((nk,ns))
        tol = np.linalg.norm(gk_old - gk_new,1)
        nitr = 0
        
        start = time.perf_counter()
        while tol > 1e-6 and nitr <= 2500:
            gk_old = gk_new.copy()
            
            # Note: using numpy.kron is much faster than using the basic kronm function 
            # that is defined here
            
            k1 = kronm(gk_old, np.ones((1,ns)))  # array of k_t+1
            k2 = linitp(k1,ssk_grid,kronm(np.ones((1,ns)),gk_old))  # array of k_t+2
            
            # array of c_t+1
            c1 = kronm(fprod(gk_old,alpha),sss[np.newaxis,:]) + (1 - delta) * k1 - k2
            c1 = np.maximum(c1,1e-6) # prohibit zero consumption

            # array of marginal productivity
            mp = 1 - delta + alpha * kronm(gk_old**(alpha - 1),sss[np.newaxis,:])
            
            gc = (beta * np.matmul(c1**(-gamma) * mp, tm1))**(-1 / gamma)
            
            gk_new = np.matmul(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:]) \
                + (1 - delta) * np.matmul(ssk[:,np.newaxis], np.ones((1,ns))) - gc
            
            gk_new = np.maximum(gk_new,ssk[0])
            gk_new = np.minimum(gk_new,ssk[-1])
            tol = np.linalg.norm(gk_old - gk_new,1)
            nitr = nitr + 1

        stop = time.perf_counter()
        t['pi'] = stop - start
        print("Elapsed time in seconds for the while loop (policy iteration) is:", \
              round(t['pi'],4))   
        print('Number of iterations of the while loop:', nitr)
        
        gk = gk_new.copy()
        gi = gk - (1 - delta) * np.dot(ssk[:,np.newaxis], np.ones((1,ns)))
        gy = gi + gc
        v = fv(gc,gk,tmt,ssk_grid,beta,gamma)
        
        
        
    elif me == 1 and vime == 0:
        # Use value function iteration with linear iterpolation
        # Use discrete maximization
        
        vnew = 1 * fu(bc,gamma) * np.ones((nk,ns)) / (1 - beta) 
        vold = np.zeros((nk,ns))
        nkd = 50 * nk
        # A denser space for k'.
        sskd = np.linspace(ssk[0],ssk[-1],nkd) # nkd x 1
        
        iv = kronv(np.ones((nk,1)), sskd[:,np.newaxis]) - \
            kronv((1 - delta) * ssk[:,np.newaxis], np.ones((nkd,1)))
        iv = kronv(np.ones((ns,1)),iv[:,np.newaxis])
        
        # 7x1, nk*nkd x 1 = nk*nkd*ns x 1
        c = kronv(sss[:,np.newaxis],kronv(fprod(ssk[:,np.newaxis],alpha),\
                                          np.ones((nkd,1)))[:,np.newaxis]) - iv
        
        del iv
        
        u = -0.5 * np.finfo(float).max * np.ones((nkd * nk * ns,))
        Ic = np.argwhere(c > 0)
        u[Ic] = fu(c[Ic],gamma)
        u = u.reshape((nkd,nk,ns),order="F")
        
        tol = np.linalg.norm(vnew - vold,1)
        nitr = 0
        
        start = time.perf_counter()
        while tol > 1e-6 and nitr <= 2500:
            vold = vnew.copy()
            vful = np.matmul(vold, tmt)
            vfuld = linitp(np.matmul(sskd[:,np.newaxis], np.ones((1,ns))),ssk_grid,vful)

            # Using the built-in numpy.kron is much faster than using the basic kronm and kronv 
            #vfuld = np.reshape(kronm(np.ones((nk,1)),vfuld),(nkd,nk,ns), order='F')
            vfuld = np.reshape(np.kron(np.ones((nk,1)), vfuld), (nkd,nk,ns), order='F') 

            #vnew = (u + beta * vfuld).max(axis=0,keepdims=True)
            #vnew = np.reshape(vnew,(nk,ns))
            #vnew = (u + beta * vfuld).max(axis=0)
            vnew = np.amax(u + beta * vfuld, axis=0)
            
            tol = np.linalg.norm(vnew - vold,1)
            nitr = nitr + 1

        stop = time.perf_counter()
        t['vi_dsc'] = stop - start
        print("Elapsed time in seconds for the while loop \
              (discrete value function iteration) is:", round(t['vi_dsc'],4))   
        print('Number of iterations of the while loop:', nitr)
        
        vful = np.dot(vnew, tmt)
        vfuld = linitp(np.dot(sskd[:,np.newaxis], np.ones((1,ns))),ssk_grid,vful)
        vfuld = np.reshape(kronm(np.ones((nk,1)), vfuld), (nkd,nk,ns), order='F')

        # Igk contains indices of gk
        v, Igk = (u + beta * vfuld).max(axis=0), (u + beta * vfuld).argmax(axis=0)
        gk = sskd[Igk]
        gi = gk - (1 - delta) * np.dot(ssk[:,np.newaxis], np.ones((1,ns)))
        gy = np.dot(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:])
        gc = gy - gi 
        
        
    else:
        # Use fminbnd to optimize
        gk = np.zeros((nk,ns))
        y = np.dot(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:]) \
        + (1 - delta) * np.dot(ssk[:,np.newaxis], np.ones((1,ns))) # Disposable income.
        
        vnew = 1 * fu(bc,gamma) * np.ones((nk,ns)) / (1 - beta) 
        vold = np.zeros((nk,ns))
        
        
        
        tol = np.linalg.norm(vnew - vold,1)
        nitr = 0
        
        start = time.perf_counter()
        while tol > 1e-6 and nitr <= 2500:
            vold = vnew.copy()
            vful = np.matmul(vold, tmt)
            for i in np.arange(0,nk):
                for j in np.arange(0,ns):
                    gk[i,j], vnew[i,j], = opt.fminbound(func=lambda k1: -fu(y[i,j]-k1,gamma)-
                                                       beta*linitp(np.array([[k1]]),\
                                                    ssk_grid,vful[:,j][:,np.newaxis]),\
                                                    x1=ssk[0],x2=min(y[i,j],ssk[-1]),\
                                                        xtol=1e-04,full_output=1)[0:2]
                    
                    # A pitfall: one needs to make sure ssk[0] is less than
                    # y[i,j]. So long as ssk contains the ergodic set of k,
                    # this condition is guaranteed.
            
            vnew = -vnew
            tol = np.linalg.norm(vnew - vold,1)
            nitr = nitr + 1

        stop = time.perf_counter()
        t['vi_ctn'] = stop - start
        print("Elapsed time in seconds for the while loop\
              (continuous value function iteration) is:", round(t['vi_ctn'],4))   
        print('Number of iterations of the while loop:', nitr)
            
        v = vnew.copy()
        gi = gk - (1 - delta) * np.dot(ssk[:,np.newaxis], np.ones((1,ns)))
        gy = np.dot(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:])
        gc = gy - gi
        
    
    return v,gk,gc,gi,gy,t
