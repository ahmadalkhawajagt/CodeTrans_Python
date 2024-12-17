"""
Hansen(1985): Continuous Dynamic Programming

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np
import matplotlib.pyplot as plt

# This is used for the root_scalar and fminbound functions
from scipy.optimize import fminbound

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)


# import the markov_approx() function
from Markov import markov_approx

# import the hp1() function
from hp1 import hp1

# needed for the maxinter() and simulation() functions, as well as the global variables
import maxinter
import simulation

# Model Parameters
alf = 0.36
beta = 0.99
a = 2
delta = 0.025
rho = 0.95
sigmae = 0.00712

# Algorithm Parameters
siyes = 1 # simulte model if equals to 1
tolv = 1e-3 # tolerance level
loadin = 0
loadsimuin = 1

# Discretize continuous shocks into Markov process
N = 7          # 7 state Markov chain
M = 3
zbar = 1       # unconditional mean of z
Tran, z, p, arho, asigma = markov_approx(rho,sigmae,M,N)
z = np.exp(z)

# Grid for the shock
lt = z.size
Pi = np.linalg.matrix_power(Tran, 10000)
Ez = 1


# Define Steady state variables
xx = (1-beta*(1-delta))/(beta*alf*Ez)
yy = ((1/beta+delta-1)/alf*(1+(1-alf)/a)-delta)*a/((1-alf)*Ez)
l_ss = xx/yy  # for this set of parameters, the steady state labor approxiamately equals to 1/3
k_ss = xx**(1/(alf-1))*l_ss
y_ss = Ez*k_ss**alf*l_ss**(1-alf)
i_ss = delta*k_ss
c_ss = y_ss-i_ss

# Define capital grid
kmin = 0.894*k_ss
kmax = 1.115*k_ss

lk = 200
gdk = (kmax-kmin)/(lk-1)
k = np.linspace(kmin,kmax,lk)

# Labor working hour
lmin = 1e-5
lmax = 1-lmin

# Initialization of the value function and policy functions
V0 = np.ones((lk,lt))
V1 = np.zeros((lk,lt))
kpr = np.zeros((lk,lt))
lpol = np.zeros((lk,lt))
polc = np.zeros((lk,lt))

# Compute the value functions and Policy Functions
iteration = 0

while np.abs(V1-V0).max(axis=0).max() > tolv:
    V0 = V1.copy()
    for t in np.arange(0,lt):
        for i in np.arange(0,lk):
            kpr[i,t]=fminbound(func=maxinter.maxinter,x1=k[0],\
                               x2=min(z[t]*k[i]**alf+(1-delta)*k[i]-0.00001,k[lk-1]),\
                                        args=(alf, k, beta, z[t], t, i, a, delta, lmin, \
                                              lmax, V0, Tran,),xtol=1e-04)
                                                             
            V1[i,t]=-maxinter.maxinter(kpr[i,t],alf, k, beta, z[t], t, i, a, delta, \
                                       lmin, lmax, V0, Tran)
            lpol[i,t] = maxinter.l
            polc[i,t] = maxinter.c
            
    iteration = iteration + 1
    print("iteration =", iteration, ", max(V1-V0) = ", np.abs((V1-V0)).max(axis=0).max())


# Make plots of value and policy functions
fig1, ax1 = plt.subplots()
ax1.plot(k, V1[:,0], 'b', label = '$z=z_{min}$')
ax1.plot(k, V1[:,-1], 'r--', label = '$z=z_{max}$')

ax1.set(xlabel='Capital Stock', title='Value Functions')
ax1.grid()
ax1.legend()

fig1.savefig("Hansen_CDP_value.jpg", dpi=800)
plt.show()


# Plot capital and consumption policy functions
fig2, axs2 = plt.subplots(2, 1)

axs2[0].plot(k, kpr[:,0], 'b', label = '$z=z_{min}$')
axs2[0].plot(k, kpr[:,-1], 'r--', label = '$z=z_{max}$')
axs2[0].plot(k, k, 'y--')
axs2[0].set(xlabel='Capital Stock', title='Capital Policy Functions')
axs2[0].grid()
axs2[0].legend()

axs2[1].plot(k, polc[:,0], 'b', label = '$z=z_{min}$')
axs2[1].plot(k, polc[:,-1], 'r--', label = '$z=z_{max}$')
axs2[1].set(xlabel='Capital Stock', title='Consumption Policy Functions')
axs2[1].grid()
axs2[1].legend()

plt.tight_layout()
plt.savefig('Hansen_CDP_policy.jpg', dpi=800)
plt.show()
plt.close(fig2)


# Plot labor policy functions
fig3, ax3 = plt.subplots()
ax3.plot(k, lpol[:,0], 'b', label = '$z=z_{min}$')
ax3.plot(k, lpol[:,-1], 'r--', label = '$z=z_{max}$')

ax3.set(xlabel='Capital Stock', title='Labor Policy Functions')
ax3.grid()
ax3.legend()

fig3.savefig("Hansen_CDP_labor.jpg", dpi=800)
plt.show()



# Simulation

if lk < 10:
    siyes = 0


if siyes == 1:
    np.random.seed(0)
    
    T = 115  # simulation time period
    n = 100  # # of simulations
    
    tind = np.zeros((n,T+1),dtype=np.int8)
    kind = np.zeros((n,T+1),dtype=np.int8)
    ktime = np.zeros((n,T+1))
    zt = np.zeros((n,T+1))
    ltime = np.zeros((n,T))
    output = np.zeros((n,T))
    invest = np.zeros((n,T))
    cons = np.zeros((n,T))
    ss_mat = np.zeros((n,6))
    cc_mat = np.zeros((n,6))
    
    for m in np.arange(0,n):
        tind[m,0] = 4
        kind[m,0] = 34
        zt[m,0] = z[tind[m,0]]
        ktime[m,0]=k[kind[m,0]]
    
        for time in np.arange(0,T):
            ktime[m,time+1]=fminbound(func=simulation.simulation,x1=k[0],\
                                      x2=min(zt[m,time]*ktime[m,time]**alf+\
                                             (1-delta)*ktime[m,time]-0.00001,k[lk-1]),\
                                        args=(k, beta, time, m, a, V0, Tran, tind, \
                                              zt[m,time], alf, ktime[m,time], lmin, \
                                              lmax, delta,),xtol=1e-04)
            
            ltime[m,time] = simulation.l
            cons[m,time] = simulation.c
            invest[m,time] = ktime[m,time+1]-(1-delta)*ktime[m,time]
            output[m,time] = cons[m,time]+invest[m,time]
            
            shock = np.random.random()
            j = 0
            while Tran[tind[m,time],0:j+1].sum() < shock:
                j = j + 1

            tind[m,time+1] = j
            zt[m,time+1] = z[j]
            
            
        logy = np.log(output[m,0:T])[np.newaxis,:]
        logc = np.log(cons[m,0:T])[np.newaxis,:]
        loginv = np.log(invest[m,0:T])[np.newaxis,:]
        logk = np.log(ktime[m,0:T])[np.newaxis,:]
        logl = np.log(ltime[m,0:T])[np.newaxis,:]
        logz = np.log(zt[m,0:T])[np.newaxis,:]
        
        
        dhp, dtr = hp1(np.concatenate((logy.T, logc.T, loginv.T, logk.T, logl.T, logz.T),\
                                      axis=1), 1600)
        ss_mat[m,:] = np.std(dhp,axis=0)*100
        Corr = np.corrcoef(dhp,rowvar=False)
        cc_mat[m,:] = Corr[:,0]
            

    stdv = np.mean(ss_mat,axis=0)
    stdv_stdv = np.std(ss_mat,axis=0)
    corr = np.mean(cc_mat,axis=0)
    corr_stdv = np.std(cc_mat,axis=0)
    
    
    print('HANSEN: std(x)/std(y) corr(x,y) for y, c, i, k, h, prod')
    print(np.concatenate((np.array([[1.36, 0.42, 4.24, 0.36, 0.7, 0.68]]).T/1.36, \
                          np.array([[1, 0.89, 0.99, 0.06, 0.98, 0.98]]).T),axis=1))
    print('std(x) std(x)/std(y)  stdv_stdv corr(x,y) corr_stdv for y, c, i, k, h, prod:')
    print(np.concatenate((stdv[:,np.newaxis], (stdv/stdv[0])[:,np.newaxis], \
                          stdv_stdv[:,np.newaxis],\
                          corr[:,np.newaxis], corr_stdv[:,np.newaxis]),axis=1))






