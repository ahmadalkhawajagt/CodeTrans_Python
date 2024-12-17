"""
user defined functions for solving the problem

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

# This is used to calculate the excution time of several loops
import time

# This is used for the root, root_scalar and minimize functions
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
    #ind1 = np.ravel_multi_index(arr_index1, (n,m), order='F') # equivalent to sub2ind in Matlab
    
    arr_index2 = (np.minimum(Ix+1,n-1).astype(int),\
                  np.matmul(np.ones((k,1)), np.arange(0,m)[np.newaxis,:]).astype(int))

    #ind2 = np.ravel_multi_index(arr_index2, (n,m), order='F') # equivalent to sub2ind in Matlab
    
    #y_gridf = y_grid.flatten('F')
    #y = (1 - w) * y_gridf[ind1] + w * y_gridf[ind2]

    y = (1 - w) * y_grid[arr_index1] + w * y_grid[arr_index2]

    return y



# utility function
def fu(c, h, a):
    u = (1 - a) * np.log(c) + a * np.log(1 - h)
    return u


# production function
def fprod(k, alpha):
    y = k**alpha
    return y

# function to determine labor supply given k',k and theta
def fh(h,rik,alpha,a):
    # rik is the ratio of investment and capital (with shock)
    y = ((1 - a) * (1 - alpha) / a + 1) * h**(1 - alpha) -\
        (1 - a) * (1 - alpha) / a * h**(-alpha) - rik
    j = ((1 - a) * (1 - alpha) / a + 1) * (1 - alpha) * h**(-alpha) +\
        (1 - a) * (1 - alpha) * alpha / a * h**(-alpha - 1)
    return y, j



def fv(gc,gh,gk,tmt,ssk_grid,beta,a):
    # solve for value function using gc, gk and tmt
    
    u = fu(gc,gh,a)  # current utility
    
    # functional equation for v
    # scipy.optimize victorizes the input and output of the function, 
    # so we must reshape the input v to its original shape, 
    # and we must flatten the output of the function
    fneq = lambda v: (u + beta * np.matmul(linitp(gk,ssk_grid,v.reshape(gk.shape)),tmt) \
                      - v.reshape(gk.shape)).flatten()
    
    v0 = 1 * u / (1 - beta)

    v = opt.root(fneq,v0,method="lm").x.reshape(gk.shape)
    # or we can use the following:
    #v = opt.fsolve(fneq,v0).reshape(gk.shape)
    return v




def indlin(x,n):
    # Compute the linearized index of an index matrix x, presuming x is
    # generated from an array w which has one more dimension of length n.
    l,k = x.shape
    
    y = x.flatten('F') + kronv(np.ones((k,1)),n * np.arange(0,l)[:,np.newaxis]) + \
        kronv(n * l * np.arange(0,k)[:,np.newaxis],np.ones((l,1)))
    
    return y.astype(int)




# Use value function interpolation or policy function iteration to solve
# Hansen's RBC model with divisible labor supply.
# Translated from Yan Liu's Matlab code, 2011.4.9
def rbcvipi(me,vime, PAR, DSS, SSK, SSS, TM):
    """ Syntax: [v,gk,gc,gi,gy,t] = bmvipi(me,vime,PAR DSS SSK SSS TM) 

    me = 1 corresponds to value function interpolation and me = 0
    corresponds to policy function iteration. vime indicates the method used
    in value function interpolation, with vime = 0 referring to discrete
    maximization and vime = 1 referring to continuous optimization
    """
    #--------------------------------------------------------------------------

    # vful_j is used to define maxv in value function iterpolation with continuous optimization
    alpha = PAR[0]
    beta = PAR[1]
    delta = PAR[2]
    gamma = PAR[3]
    a = PAR[4]
    bk = DSS[0]  # use the syntax b* to denote steady state value
    bc = DSS[1]
    bh = DSS[2]
    ssk = SSK  # nkx1
    nk = ssk.size
    sss = SSS     # ns x1
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
                                                       np.ones((1,ns)))\
        + bk * np.matmul(np.ones((nk,1)), (0.1 * (sss[np.newaxis,:] - 1) + 1))     
        gk_new = np.maximum(gk_new,ssk[0])
        gk_new = np.minimum(gk_new,ssk[-1])
        gh_new = -0.5 * alpha * delta * bk**(-alpha) /\
        ((1 - alpha) * ((1 - a) / a * (1 - alpha) + 1) * bh**(-alpha) + \
        alpha * (1 - alpha) * (1 - a) / a * bh**(-alpha - 1)) * \
        (ssk[:,np.newaxis] * np.ones((1,ns)) - bk) + bh * np.ones((nk,1)) * \
            (0.1 * (sss[np.newaxis,:] - 1) + 1)
        # This choice is motivated by the approximation around steady state.
        
        
        gk_old = np.zeros((nk,ns))
        gh_old = np.zeros((nk,ns))
        tol = max(np.linalg.norm(gk_old - gk_new,1),np.linalg.norm(gh_old - gh_new,1))
        nitr = 0
        
        start = time.perf_counter()
        while tol > 1e-6 and nitr <= 2500:
            gk_old = gk_new.copy()
            gh_old = gh_new.copy()
            
            # Note: using numpy.kron is much faster than using the basic kronm function 
            # that is defined here
            
            k1 = kronm(gk_old, np.ones((1,ns)))  # array of k_t+1
            k2 = linitp(k1,ssk_grid,kronm(np.ones((1,ns)),gk_old))  # array of k_t+2
            
            h1 = kronm(np.ones((1,ns)),linitp(gk_old,ssk_grid,gh_old))
            
            c1 = kronm(fprod(gk_old,alpha),sss[np.newaxis,:])*(h1**(1 - alpha)) +\
                (1 - delta) * k1 - k2 # array of c_t+1
            c1 = np.maximum(c1,1e-3) # prohibit zero consumption
            c1 = np.minimum(c1,5)

            # array of marginal productivity
            mp = 1 - delta + alpha * kronm(gk_old**(alpha - 1),\
                                           sss[np.newaxis,:])*(h1**(1 - alpha)) 
            
            gc = (beta * np.matmul(c1**(-gamma) * mp, tm1))**(-1 / gamma)
            
            for i in np.arange(0,nk):
                for j in np.arange(0,ns):
                    gh_new[i,j] = opt.root_scalar(lambda h: (h**alpha / (1 - h) - \
                        (1 - a) * (1 - alpha) / a / gc[i,j] * sss[j] * ssk[i]**alpha),\
                                              bracket=[1e-4,1 - 1e-4]).root
                    
            gh_new = np.minimum(gh_new,0.5)
            gh_new = np.maximum(gh_new,0.1)
            
            gk_new = np.matmul(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:]) * \
                (gh_new**(1 - alpha)) \
            + (1 - delta) * np.matmul(ssk[:,np.newaxis], np.ones((1,ns))) - gc
            gk_new = np.maximum(gk_new,ssk[0])
            gk_new = np.minimum(gk_new,ssk[-1])
            
            tol = max(np.linalg.norm(gk_old - gk_new,1),np.linalg.norm(gh_old - gh_new,1))
            nitr = nitr + 1

        stop = time.perf_counter()
        t['pi'] = stop - start
        print("Elapsed time for the while loop (policy iteration) is:", \
              round(t['pi'],4), "seconds")   
        print('Number of iterations of the while loop:', nitr)
        
        gk = gk_new.copy()
        gh = gh_new.copy()
        gi = gk - (1 - delta) * np.matmul(ssk[:,np.newaxis], np.ones((1,ns)))
        gy = gi + gc
        v = fv(gc,gh,gk,tmt,ssk_grid,beta,a)
        
     
    elif me == 1 and vime == 0:
        # Use value function iteration with linear iterpolation
        # Use discrete maximization
        
        vnew = 1 * fu(bc,bh,a) * np.ones((nk,ns)) / (1 - beta) 
        vold = np.zeros((nk,ns))
        nkd = 50 * nk
        # A denser space for k'.
        sskd = np.linspace(ssk[0],ssk[-1],nkd) # nkd x 1
        
        iv = kronv(np.ones((nk,1)), sskd[:,np.newaxis]) - \
            kronv((1 - delta) * ssk[:,np.newaxis], np.ones((nkd,1)))
        iv = kronv(np.ones((ns,1)),iv[:,np.newaxis])
        
        
        # for each combination of k',k and theta, solve for h
        h = np.zeros((nkd * nk * ns,1))   # heuristic state space of h
        prodk = kronv(sss[:,np.newaxis],kronv(fprod(ssk[:,np.newaxis],alpha),\
                                              np.ones((nkd,1)))[:,np.newaxis])
        rik = iv / prodk
        
        start = time.perf_counter()
        for i in np.arange(0,nkd * nk * ns):
            fh_i = lambda x: fh(x,rik[i],alpha,a)[0]
            h[i] = opt.root_scalar(fh_i, bracket=[0.001,2]).root

        th = time.perf_counter() - start
        print("Time for computing h is:", round(th,4),"seconds")
        h = np.minimum(h,1 - 1e-6)
        h = np.maximum(h,0)   # truncate h at 0 and 1
        
        c = prodk*np.squeeze(h)**(1 - alpha)-iv
        Ic = np.argwhere(c > 0)
        
        u = -0.5 * np.finfo(float).max * np.ones((nkd * nk * ns,1))
        u[Ic] = fu(c[:,np.newaxis][Ic],h[Ic],a)
        
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
        print("Elapsed time for the while loop (discrete value function iteration) is:", \
              round(t['vi_dsc'],4),"seconds")   
        print('Number of iterations of the while loop:', nitr)
        
        vful = np.matmul(vnew, tmt)
        vfuld = linitp(np.matmul(sskd[:,np.newaxis], np.ones((1,ns))),ssk_grid,vful)
        vfuld = np.reshape(kronm(np.ones((nk,1)), vfuld), (nkd,nk,ns), order='F')

        v, Igk = (u + beta * vfuld).max(axis=0), (u + beta * vfuld).argmax(axis=0) 
        gk = sskd[Igk] # Igk contains indices of gk
        
        Iopt = np.reshape(indlin(Igk,nkd),(nk,ns),order='F')

        # compute the linearized index for the optimal choice array of the
        # form k.T by k by theta
        gi = iv[Iopt]
        gh = h.squeeze()[Iopt]
        gc = c[Iopt]
        gy = gc + gi
        
        
    else:
        # Use fmincon to optimize
        
        gk = (1 - (1 - alpha) * delta) * np.matmul((ssk - bk)[:,np.newaxis], np.ones((1,ns)))\
        + bk * np.matmul(np.ones((nk,1)), (0.1 * (sss[np.newaxis,:] - 1) + 1))
        # gk = np.maximum(gk,ssk[0])
        # gk = np.minimum(gk,ssk[-1])
        
        gh = -0.5 * alpha * delta * bk**(-alpha) /\
        ((1 - alpha) * ((1 - a) / a * (1 - alpha) + 1) * bh**(-alpha) + \
        alpha * (1 - alpha) * (1 - a) / a * bh**(-alpha - 1)) * \
        (ssk[:,np.newaxis] * np.ones((1,ns)) - bk) + \
            bh * np.ones((nk,1)) * (0.1 * (sss[np.newaxis,:] - 1) + 1)
                
        gc = np.matmul(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:]) * (gh**(1 - alpha))+\
        (1 - delta) * np.matmul(ssk[:,np.newaxis], np.ones((1,ns))) - gk
        gc = np.maximum(gc,0.1)
        
        vnew = fv(gc,gh,gk,tmt,ssk_grid,beta,a)
        vold = np.zeros((nk,ns))

        ub = np.array([ssk[-1],0.8])
        lb = np.array([ssk[0] - 5,0.1])
        bnds = ((lb[0],ub[0]),(lb[1],ub[1]))
                
        tol = np.linalg.norm(vnew - vold,1)
        nitr = 0
        start = time.perf_counter()
        while tol > 1e-4 and nitr <= 1500:
            vold = vnew.copy()
            vful = np.matmul(vold, tmt)
            
            nitr = nitr + 1
            startfor = time.perf_counter()
            for i in np.arange(0,nk):
                for j in np.arange(0,ns):

                    maxv = lambda x: -fu(sss[j] * ssk[i]**alpha * x[1] \
                                         + (1 - delta) * ssk[i] -x[0],\
                                         x[1]**(1 / (1 - alpha)),a)-\
                    beta*linitp(np.array([[x[0]]]),ssk_grid,vful[:,j][:,np.newaxis])
                    
                    A = np.array([[1,-sss[j] * ssk[i]**alpha]])
                    b = np.array([(1 - delta) * ssk[i] - 0.1])
                    cons = [{"type": "ineq", "fun": lambda x: A @ x - b}]
                    
                    result = opt.minimize(maxv,x0=(gk[i,j],gh[i,j]**(1 - alpha)),\
                                          method='trust-constr',\
                                         bounds=bnds,constraints=cons)
                    
                    # Note: there is a problem with the minimize method. 
                    # Some of the values that the minimize method chooses
                    # create a concumption value that is negative, 
                    # which created an error when the log is taken inside
                    # the fu (utility) function

                    #,args=(sss,ssk,alpha,beta,ssk_grid,vful,i,j,)
                                        
                    g = result.x
                    vnew[i,j] = result.fun
                    gk[i,j] = g[0]
                    gh[i,j] = g[1]**(1 / (1 - alpha))
            

            stopfor = time.perf_counter()
            print("Elapsed time for the for loops for iteration",\
                  '\033[1m' + str(nitr) + '\033[0m',\
                  "is",round(stopfor-startfor,4),"seconds")
            vnew = -vnew
            tol = np.linalg.norm(vnew - vold,1)
            

        stop = time.perf_counter()
        t['vi_ctn'] = stop - start
        print("Elapsed time for the while loop (continuous value function iteration) is:",\
              round(t['vi_ctn'],4),"seconds")   
        print('Number of iterations of the while loop:', nitr)
            
        v = vnew.copy()
        gi = gk - (1 - delta) * np.dot(ssk[:,np.newaxis], np.ones((1,ns)))
        gy = np.dot(fprod(ssk[:,np.newaxis],alpha), sss[np.newaxis,:])
        gc = gy - gi
        
    
    return v,gk,gh,gc,gi,gy,t