"""
This program solves the stochastic growth model with linear quadratic dynamic programming.

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
alf = 0.33
beta = 0.96
rho = 0.95
delta = 0.10
tolv = 1e-07
sigmae = 0.007



# Steady state
zs = 0
ks = (np.exp(zs)*alf*beta/(1-beta*(1-delta)))**(1/(1-alf))
ins = delta*ks
cs = np.exp(zs)*ks**alf-ins



# Construct the quadratic expansion of the utility function
R = np.log(cs)

DJ = np.zeros((3,1))
DJ[0,0] = (np.exp(zs)*ks**alf)/cs #Jz
DJ[1,0] = (np.exp(zs)*alf*ks**(alf-1))/cs #Jk
DJ[2,0] = (-1)/cs #Jx

Hzz = ((np.exp(zs)*ks**alf)*cs - (np.exp(zs)*ks**alf)**2 )/(cs**2)
Hkk = (((np.exp(zs)*alf*(alf-1)*ks**(alf-2))*cs)-(np.exp(zs)*alf*ks**(alf-1))**2 )/(cs**2)
Hxx = (-1)/(cs**2)
Hzk = (((np.exp(zs)*alf*ks**(alf-1))*cs) - \
       (np.exp(zs)*ks**alf*np.exp(zs)*alf*ks**(alf-1)))/(cs**2)
Hzx = (np.exp(zs)*ks**alf)/(cs**2)
Hkx = (np.exp(zs)*alf*ks**(alf-1))/(cs**2)

DH = np.array([[Hzz, Hzk, Hzx],
    [Hzk, Hkk, Hkx],
    [Hzx, Hkx, Hxx]])

S = np.array([[zs, ks]])
C = np.array([[ins]])



# Define input matrix B
B = np.array([[1, 0, 0, 0],
    [0, rho, 0, 0],
    [0, 0, 1-delta, 1]])

# Define the variance covariance matrix
Sigma = np.array([[0,0,0],[0,sigmae**2,0],[0,0,0]])


P,J,d = riccati1(R,DJ,DH,S,C,B,Sigma,beta)


print(' The optimal value function is [1 z s]P0[1; z; s]+d, where P and d are given by:')
print(P)
print(round(d,4))

print(' The policy function is x=J[1; z; s] where J is:')
print(J)



# Simulation of the model
T = 115
N = 100
ss = np.zeros((N,4))
cc = np.zeros((N,4))
rng = np.random.Generator(np.random.MT19937())

for j in np.arange(0,N):
    r = rng.standard_normal((T+1,1))
    z = np.ones((T+1,1))
    z[0] = 0
    k = np.zeros((T+1,1))
    k[0] = ks
    i = np.zeros((T,1))
    c = np.zeros((T,1))
    y = np.zeros((T,1))

    for t in np.arange(0,T):
        y[t] = np.exp(z[t])*k[t]**alf
        i[t] = J@np.array([[1],z[t],k[t]])
        c[t] = y[t]-i[t]
        k[t+1] = (1-delta)*k[t] + i[t]
        z[t+1] = rho*z[t]+sigmae*r[t]


    z = z[0:T]
    k = np.log(k[0:T])
    y = np.log(y[0:T])
    c = np.log(c)
    i = np.log(i)

    dhp, dtr = hp1(np.concatenate((y, c, i, k),axis=1),1600)
    ss[j,:] = np.std(dhp,axis=0,ddof=1)*100
    Corr = np.corrcoef(dhp,rowvar=False)
    cc[j,:] = Corr[:,0]

stdv = np.mean(ss,axis=0)
corr = np.mean(cc,axis=0)

print('std(x) std(x)/std(y) corr(x,y) for y, c, i, k:')
print(np.concatenate((stdv[:,np.newaxis], (stdv/stdv[0])[:,np.newaxis], \
                      corr[:,np.newaxis]),axis=1))






