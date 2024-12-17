"""
This program solves the stochastic growth model labor leisure choice 
with linear quadratic dynamic programming.

Translated from Eva Carceles-Poveda's (2003) MATLAB codes
"""

# Importing packages
import numpy as np

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)


# import the riccati1() function
from riccati1 import riccati1

# import the hp1() function
from hp1 import hp1


# Model Parameters
beta    = 0.99     # Discount factor
delta   = 0.025    # Depreciations
alpha   = 0.36     # Capital's share
rho     = 0.95     # Autocorrelation of shock
gamma   = 1        # One plus the quarterly growth rate of technology
sigmae  = 0.00712


# Steady state
zbar    =  1
hbar    =  0.3
kbar    =  hbar * (((gamma/beta - (1- delta )) / alpha )**(1/(alpha - 1)))
ibar    =  (gamma - 1 + delta) * kbar
ybar    =  (kbar**alpha)*(hbar**(1-alpha))
cbar    =  ybar - ibar
prodbar =  ybar / hbar
Rbar    =  alpha*(kbar**(alpha-1))*(hbar**(1-alpha))
wbar    =  (1-alpha)*(kbar**alpha)*(hbar**(-alpha))
a       =  (1-hbar)*wbar/cbar
print('The steady state values of z, y, i, c, k and h are:',np.array([zbar, ybar, ibar, cbar, kbar, hbar]))


# Obtain a quadratic approximation of the return function
Ubar    =  np.log(cbar) + a*np.log(1 - hbar)

# Construct the quadratic expansion of the utility function
Uz  = (kbar**alpha)*(hbar**(1-alpha))/cbar
Uk  = alpha*zbar*(kbar**(alpha-1))*(hbar**(1-alpha))/cbar
Ui  = -1/cbar
Uh  = (1-alpha)*zbar*(kbar**alpha)*(hbar**(-alpha))/cbar -a/(1-hbar)
DJ  = np.array([[Uz],[Uk],[Ui],[Uh]])

c2  = cbar**2
Ukk = ((alpha-1)*alpha*zbar*(kbar**(alpha -2))*(hbar**(1-alpha))*cbar \
    -(alpha*zbar*(kbar**(alpha-1))*(hbar**(1-alpha))) \
    *(alpha*zbar*(kbar**(alpha-1))*(hbar**(1-alpha))) )/c2

Ukz = ((alpha*(kbar**(alpha-1))*(hbar**(1-alpha))*cbar \
    -(alpha*zbar*(kbar**(alpha-1))*(hbar**(1-alpha))) \
    *(kbar**alpha)*(hbar**(1-alpha))) )/c2

Uki = alpha*zbar*(kbar**(alpha-1))*(hbar**(1-alpha))/c2

Ukh = (((1-alpha)*alpha*zbar*(kbar**(alpha-1))*(hbar**(-alpha)))*cbar \
    -(alpha*zbar*(kbar**(alpha-1))*(hbar**(1-alpha))) \
    *((1-alpha)*zbar*(kbar**alpha)*(hbar**(-alpha))))/c2

Uzz = -((kbar**(alpha))*(hbar**(1-alpha)) \
    *(kbar**(alpha))*(hbar**(1-alpha)))/c2

Uzi = (kbar**(alpha))*(hbar**(1-alpha))/c2

Uzh = ((1-alpha)*(kbar**(alpha))*(hbar**(-alpha))*cbar \
    -(kbar**(alpha))*(hbar**(1-alpha)) \
    *((1-alpha)*zbar*(kbar**(alpha))*(hbar**(-alpha))))/c2

Uii = -1/c2
Uih = ((1-alpha)*zbar*(kbar**(alpha))*(hbar**(-alpha)))/c2

Uhh = ((-alpha*(1-alpha)*zbar*(kbar**(alpha))*(hbar**(-alpha-1)))*cbar \
    - ((1-alpha)*zbar*(kbar**(alpha))*(hbar**(-alpha))) \
    *((1-alpha)*zbar*(kbar**(alpha))*(hbar**(-alpha))))/c2 -a/((1-hbar)**2);

DH = np.array([[Uzz, Ukz, Uzi, Uzh], [Ukz, Ukk, Uki, Ukh], [Uzi, Uki, Uii, Uih], [Uzh, Ukh, Uih, Uhh]])

S = np.array([[zbar], [kbar]])
C = np.array([[ibar], [hbar]])

B = np.array([[1, 0, 0, 0, 0], [1-rho, rho, 0, 0, 0], [0 , 0, 1-delta, 1, 0]])

Sigma = np.array([[0, 0, 0], [0, sigmae**2, 0], [0, 0, 0]])



# Calculate the value function
P,J,d = riccati1(Ubar,DJ,DH,S,C,B,Sigma,beta)



print(' The optimal value function is [1 z s]P0[1; z; s]+d, where P and d are given by:')
print(P)
print(round(d,4))

print(' The policy function is x=J[1; z; s] where J is:')
print(J)



# simulate the artificial economy
T = 115
N = 100
ss_mat = np.zeros((N,6))
cc_mat = np.zeros((N,6))
rng = np.random.Generator(np.random.MT19937())

for j in np.arange(0,N):
    r = rng.standard_normal((T+1,1))
    z = np.ones((T+1,1))
    z[0] = 1
    k = np.zeros((T+1,1))
    k[0] = kbar
    i = np.zeros((T,1))
    c = np.zeros((T,1))
    y = np.zeros((T,1))
    h = np.zeros((T,1))
    prod = np.zeros((T,1))

    for t in np.arange(0,T):
        i[t] = J[0,:]@np.array([[1],z[t],k[t]])
        h[t] = J[1,:]@np.array([[1],z[t],k[t]])
        y[t] = z[t]*k[t]**alpha*h[t]**(1-alpha)

        c[t] = y[t]-i[t]
        k[t+1] = (1-delta)*k[t] + i[t]
        z[t+1] = 1-rho+rho*z[t]+sigmae*r[t]
        prod[t] = y[t]/h[t]
        

    z = z[0:T]
    k = np.log(k[0:T])
    y = np.log(y[0:T])
    c = np.log(c)
    i = np.log(i)
    h = np.log(h[0:T])
    prod = np.log(prod)

    dhp, dtr = hp1(np.concatenate((y, i, c, k, h, prod),axis=1),1600)
    ss_mat[j,:] = np.std(dhp,axis=0,ddof=1)*100
    Corr = np.corrcoef(dhp,rowvar=False)
    cc_mat[j,:] = Corr[:,0]

std = np.mean(ss_mat,axis=0)
corr = np.mean(cc_mat,axis=0)

print('HANSEN: std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
print(np.concatenate((np.array([[1.36, 4.24, 0.42, 0.36, 0.7, 0.68]]).T/1.36, \
                      np.array([[1, 0.99, 0.89, 0.06, 0.98, 0.98]]).T),axis=1))
print('std(x) std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
print(np.concatenate((std[:,np.newaxis], (std/std[0])[:,np.newaxis], \
                      corr[:,np.newaxis]),axis=1))





