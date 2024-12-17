"""
Hansen(1985): Discrete Dynamic Programming

Translated from Eva Carceles-Poveda's MATLAB codes
"""


# Importing packages
import numpy as np
import matplotlib.pyplot as plt

# This is used for the root_scalar function
from scipy.optimize import root_scalar

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)


# import the markov_approx() function
from Markov import markov_approx

# import the acm() function
from acm import acm

# import the hp1() function
from hp1 import hp1

# import the lfoc() function
from lfoc import lfoc


# Model Parameters
alf = 0.36
beta = 0.99
a = 2
delta = 0.025
rho = 0.95
sigmae = 0.00712


# Algorithm Parameters
siyes = 1       # simulte model if equals to 1
loadin = 0      # load policy functions if equals 1
loadsimuin = 0  # load simulation results if equals to 1
stationary = 1  # finds stationary distribution for capital
tolv = 1e-12    # tolerance level



# Discretize continuous shocks into Markov process
N = 7          # 7 state Markov chain
m = 3
zbar = 1       # unconditional mean of z
z,Tran = acm(0,rho,sigmae,N,1)[0:2]  # Adda-Cooper Shocks
#Tran, z, p, arho, asigma = markov_approx(rho,sigmae,m,N) # Shocks by Tauchen method 
z=np.exp(z)

lt = z.size # lt is the same size as N.
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
gk = lk*lk


#--------------------------------------------------------------------------
# Compute objective function
#--------------------------------------------------------------------------
c = np.zeros((gk,lt))
l = np.zeros((gk,lt))
lmin = 1e-5
lmax = 1-lmin


for t in np.arange(0,lt):
    for i in np.arange(0,lk):  # today's capital
        for j in np.arange(0,lk):  # tomorrow's capital
            # this if-elseif loop helps to set up reasonable range of l so
            # that we will not get NAN
            if k[j]-(1-delta)*k[i]> 0 and \
                ((k[j]-(1-delta)*k[i])/(z[t]*k[i]**alf))**(1/(1-alf)) < lmax:
                L = root_scalar(lfoc, args=(z[t],k[i],k[j],alf,a,delta), \
                          bracket=[((k[j]-(1-delta)*k[i])/\
                                    (z[t]*k[i]**alf))**(1/(1-alf)),lmax]).root
            elif k[j]-(1-delta)*k[i]<0:
                L = root_scalar(lfoc, args=(z[t],k[i],k[j],alf,a,delta), \
                                bracket=[lmin,lmax]).root

            if L < 0:
                L=lmin
            elif L > 1:
                L=lmax

            l[i*lk+j,t] = L
            c[i*lk+j,t] = z[t]*(k[i]**alf)*L**(1-alf)+(1-delta)*k[i]-k[j]
            if c[i*lk+j,t] < 0:
                c[i*lk+j,t] = 1e-12

U = np.log(c)+a*np.log(1-l)


# Initialization of the value function
V0 = np.ones((lk,lt))
V1 = np.zeros((lk,lt))

# Iterate on the value function
while np.linalg.norm(V1-V0) > tolv: 
    V0=V1.copy()
    for t in np.arange(0,lt):
        for i in np.arange(0,lk):
            V1[i,t]=(U[i*lk:(i+1)*lk,t]+beta*np.matmul(V0,Tran.T[:,t])).max()


# Policy functions
optk = np.zeros((lk,lt)).astype(int)
lpol = np.zeros((lk,lt))
for t in np.arange(0,lt):
    for i in np.arange(0,lk):
        optk[i,t] = (U[i*lk:(i+1)*lk,t]+beta*np.matmul(V0,Tran.T[:,t])).argmax()
        lpol[i,t]=l[i*lk+optk[i,t],t]

polk = k[optk]
poly = np.matmul(np.ones((lk,1)),z[np.newaxis,:])*np.matmul(k[:,np.newaxis]**alf,\
                                                            np.ones((1,lt)))*(lpol**(1-alf))
poli = polk-(1-delta)*np.matmul(k[:,np.newaxis],np.ones((1,lt)))
polc = poly-poli



# Plot value function
fig1, ax1 = plt.subplots()
ax1.plot(k, V1[:,0], 'b', label = '$z=z_{min}$')
ax1.plot(k, V1[:,-1], 'r--', label = '$z=z_{max}$')

ax1.set(xlabel='Capital Stock', title='Value Functions')
ax1.grid()
ax1.legend()

fig1.savefig("Hansen_DDP_value.jpg", dpi=800)
plt.show()



# Plot capital and consumption policy functions
fig2, axs2 = plt.subplots(2, 1)

axs2[0].plot(k, polk[:,0], 'b', label = '$z=z_{min}$')
axs2[0].plot(k, polk[:,-1], 'r--', label = '$z=z_{max}$')
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
plt.savefig('Hansen_DDP_policy.jpg', dpi=800)
plt.show()
plt.close(fig2)

# Plot investment and labor policy functions
fig3, axs3 = plt.subplots(2, 1)

axs3[0].plot(k, poli[:,0], 'b', label = '$z=z_{min}$')
axs3[0].plot(k, poli[:,-1], 'r--', label = '$z=z_{max}$')
axs3[0].set(xlabel='Capital Stock', title='Investment Policy Functions')
axs3[0].grid()
axs3[0].legend()

axs3[1].plot(k, lpol[:,0], 'b', label = '$z=z_{min}$')
axs3[1].plot(k, lpol[:,-1], 'r--', label = '$z=z_{max}$')
axs3[1].set(xlabel='Capital Stock', title='Labor Policy Functions')
axs3[1].grid()
axs3[1].legend()

plt.tight_layout()
plt.savefig('Hansen_DDP_policy.jpg', dpi=800)
plt.show()
plt.close(fig3)


#--------------------------------------------------------------------------
# Monte-Carlo Simulation
#--------------------------------------------------------------------------
if lk < 10:
    siyes = 0


if siyes == 1:
    np.random.seed(0)
    
    T = 115  # simulation time period
    n = 100  # # of simulations
    
    tind = np.zeros((n,T+1),dtype=np.int8)
    kopt = np.zeros((n,T+1))
    zt = np.zeros((n,T+1))
    lopt = np.zeros((n,T))
    output = np.zeros((n,T))
    invest = np.zeros((n,T))
    cons = np.zeros((n,T))
    ss_mat = np.zeros((n,6))
    cc_mat = np.zeros((n,6))
    
    for i in np.arange(0,n):
        tind[i,0] = 4
        zt[i,0] = z[tind[i,0]]
        indk = round(lk/2)-1
        kopt[i,0] = polk[indk,0]
    
        for t in np.arange(0,T):
            indk = optk[indk,tind[i,t]]
            kopt[i,t+1] = polk[indk,tind[i,t]]
            lopt[i,t] = lpol[indk,tind[i,t]]
            output[i,t] = zt[i,t]*(kopt[i,t]**alf*lopt[i,t]**(1-alf))
            invest[i,t] = kopt[i,t+1]-(1-delta)*kopt[i,t]
            cons[i,t] = polc[indk,tind[i,t]]
            
            shock = np.random.random()
            j = 0
            while Tran[tind[i,t],0:j+1].sum() < shock:
                j = j + 1

            tind[i,t+1] = j
            zt[i,t+1] = z[j]
            
            
        logy = np.log(output[i,0:T])[np.newaxis,:]
        logc = np.log(cons[i,0:T])[np.newaxis,:]
        loginv = np.log(invest[i,0:T])[np.newaxis,:]
        logk = np.log(kopt[i,0:T])[np.newaxis,:]
        logl = np.log(lopt[i,0:T])[np.newaxis,:]
        logz = np.log(zt[i,0:T])[np.newaxis,:]
        
        
        dhp, dtr = hp1(np.concatenate((logy.T, logc.T, loginv.T, logk.T, logl.T, logz.T),\
                                      axis=1), 1600)
        ss_mat[i,:] = np.std(dhp,axis=0)*100
        Corr = np.corrcoef(dhp,rowvar=False)
        #print(ss_mat[i,:].shape)
        #print(Corr.shape)
        cc_mat[i,:] = Corr[:,0]
            

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

