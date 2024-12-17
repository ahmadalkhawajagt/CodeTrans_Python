"""
This program solves the model of Aiyagari(94)

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the markov approximation function 
from Markov import markov_approx


# Parameter values
beta = 0.96
timepref = 1/beta-1
mu = 3
theta = 0.36
delta = 0.08
rho = 0.9
sigma = 0.4
sigmain = sigma*np.sqrt(1-rho**2)


# N state markov chain
N = 7

Pi,logs,invdist = markov_approx(rho,sigmain,3,N)[0:3] # Pi gives Pij=prob(j given i)
s = np.exp(logs)
labor = s@invdist
lt = max(s.shape)
Pii = Pi.T # Pij=prob(i given j)



# Algorithm parameters
tolv = 1e-07
tolr = 1e-04
tola = 1e-03

ini = 1
if ini:
    r = 0.04
else:
    r = 0.02298754441505 #394

# Initial interest rate and grid for the states
k = ((r+delta)/(theta*labor**(1-theta)))**(1/(theta-1))
w = (1-theta)*k**theta*labor**(-theta)

# asset limit and gridb=0;
b = 0
if r<=0:
    phi = b
else:
    phi  = min(b, w*s[0]/r)

minkap = -phi
maxkap = 16.00
inc = 0.2
#inc = 0.405
kgrid  = np.arange(minkap,maxkap+inc,inc)
lk = max(kgrid.shape)

# Main loop
testr = 1
testa = 1
iter1 = 0
while testr > tolr or testa > tola:
    iter1 = iter1+1
    print("Iteration", iter1, ":")
    oldr = r
    
    k = ((r+delta)/(theta*labor**(1-theta)))**(1/(theta-1))
    w = (1-theta)*k**theta*labor**(-theta)
    
    # iterate on the value function and compute the optimal policy
    c = np.zeros((lk*lk,lt))
    u = np.zeros((lk*lk,lt))
    for t in np.arange(0,lt):
        for i in np.arange(0,lk):
            for j in np.arange(0,lk):
                c[i*lk+j,t] = (1+r)*kgrid[i]+w*s[t]-kgrid[j]
                if c[i*lk+j,t]<0:
                    c[i*lk+j,t] = 1e-07
    
    if mu==1:
        u = np.log(c)
    else:
        u = (c**(1-mu)-1)/(1-mu)
    
    
    V0 = np.ones((lk,lt))
    V1 = np.zeros((lk,lt))
    optim = np.zeros((lk,lt),dtype=int)
    
    # Value function iteration
    iter2 = 0
    while np.linalg.norm(V1-V0) > tolv:
        #print(iter2, np.linalg.norm(V0-V1))
        iter2 = iter2+1
        V0 = V1.copy()
        for j in np.arange(0,lt):
            for i in np.arange(0,lk):
                f = u[i*lk:(i+1)*lk,j]+np.matmul(beta*V0,Pii[:,j])
                V1[i,j] = f.max()
                optim[i,j] = f.argmax()
                

    # Policy function conditional on shock
    polk = kgrid[optim]   
    
    # Calculate the invariant distribution
    gmat = np.zeros((lk,lk,lt))
    trans = np.zeros((lk*lt,lk*lt))
    for j in np.arange(0,lt):
        for i in np.arange(0,lk):
            gmat[i,optim[i,j],j] = 1;

        trans[j*lk:(j+1)*lk,:] = np.kron(Pi[j,:],gmat[:,:,j])
    
    trans = trans.T
    probst = (1/(lt*lk))*np.ones((lt*lk,1))
    
    test = 1
    while test > 10**(-5):
        probst1 = trans@probst
        test = np.abs(probst1-probst).max()
        probst = probst1.copy()
    
    
    # change the dimension of polk
    kk = polk.flatten(order='F')
    meank = probst.T@kk
    
    # Update r if necessary
    rstar=theta*((meank)**(theta-1))@(labor**(1-theta))-delta
    testr = np.abs(r-rstar)
    testa = np.abs(k-meank)
    
    if iter1==1:
        if rstar>(1/beta)-1:
            rstar=(1/beta)-1
        
        rhigh=max([r,rstar]);
        rlow=min([r,rstar]);
    elif meank > k:
        print('saving too much (meank>k), so reducing r!')
        rhigh=r
    else:
        print('saving too little (meank<k), so increasing r!')
        rlow=r
    
    r=rhigh*0.5+rlow*0.5
    print('testr testa oldr rstar')
    print(np.around([testr, testa[0], oldr, rstar],4))

print('Final solution: r, k, meank, s, testr testa')
print(np.around([r, k[0], meank[0], (delta*meank/((meank**theta)*labor**(1-theta)))[0], testr, testa[0]],4))



