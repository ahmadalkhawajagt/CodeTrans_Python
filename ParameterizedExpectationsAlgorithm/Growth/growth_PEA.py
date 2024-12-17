"""
The following program solves the stochastic growth model with simulations PEA.
It uses a non-linear least squares routine to minimize the sum of squared errors.

Translated from Eva Carceles-Poveda's (2003) MATLAB codes
"""

# Importing packages
import numpy as np

import matplotlib.pyplot as plt

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the nlls() function
from nlls import nlls

# import the hp1() function
from hp1 import hp1


# Parameters
bet = 0.99
alf = 0.36
delta = 0.025
T = 10000
sig = 1
rhoz = 0.95
sigepsz = 0.01
lam = 0



# load shocks

# from numpy (.npy) file
with open('zz.npy', 'rb') as f:
    z = np.load(f)

# or from text (.txt) file
#with open('zz.txt', 'rb') as f:
#    z = np.loadtxt(f)


# Steady state
zs = 1
ks = ((1-bet*(1-delta))/(bet*alf))**(1/(alf-1))
ys = zs*(ks**alf)
ins = delta*ks
cs = ys-ins

print('Steady State:', ks, ys, cs, ins)



# Initial parameters for the PEA functions
bita = np.zeros((3,1))
#with open('bita.npy', 'rb') as f:
#    bita = np.load(f)
bita0 = np.ones((3,1))
bita = (bita-lam)/(1-lam)

# Define matrices
x = np.zeros((T+1,1))
c = np.zeros((T+1,1))
k = np.zeros((T+1,1))
inv = np.zeros((T+1,1))
Pea = np.zeros((T+1,1))

etol = 1e-03
it = 0

while np.abs(bita-bita0).max() > etol:
    it = it+1
    bita = lam*bita0+(1-lam)*bita
    bita0 = bita.copy()
    
    # Simulation for given parameters and no asset constraints

    k[0]=ks
    c[0]=cs
    inv[0]=ins

    for i in np.arange(1,T+1):
        Pea[i]=np.exp(bita[0]+bita[1]*np.log(z[i])+bita[2]*np.log(k[i-1]))
        c[i]=(bet*Pea[i])**(-1/sig)
        inv[i]=z[i]*(k[i-1]**alf)-c[i]
        k[i]=inv[i]+(1-delta)*k[i-1]
    
    E3=((c[2:T+1]**(-sig))*(alf*z[2:T+1][:,np.newaxis]*(k[1:T]**(alf-1))+(1-delta)))
    
    # calculating the minimizing parameters
    bita = nlls(E3,np.log(z[1:T])[:,np.newaxis],np.log(k[0:T-1]),T,bita0)
    eva = np.array([np.mean(k), np.mean(c), np.mean(inv)])
    print("Means:", eva)
    print("iteration = ", it, ", error = ", np.abs(bita-bita0).max(), sep='')
    #with open('bita.npy', 'wb') as f:
    #    np.save(f, bita)
    


print(bita)



###########################
# Make impulse responses
###########################
print(' ')
while True:
    try:
        NS = int(input('Enter number of periods for IRFs: '))
    except ValueError:
        print("Please enter a valid integer")
        continue
    else:
        if NS < 2:
            print("Please enter a positive integer greater than 1")
            continue
        else:
            print(f'You entered: {NS}')
            break

NS             = NS+1
eps            = np.zeros((NS,1))
eps[1]         = 1
shock          = np.zeros((NS,1))
shock[0]       = 1
Pea            = np.zeros((NS,1))

c              = np.zeros((NS,1)); c[0] = cs
k              = np.zeros((NS,1)); k[0] = ks
y              = np.zeros((NS,1)); y[0] = ys
inv            = np.zeros((NS,1)); inv[0] = ins

for i in np.arange(1,NS):
    shock[i] = np.exp(rhoz*np.log(shock[i-1])+eps[i])
    Pea[i] = np.exp(bita[0]+bita[1]*np.log(shock[i])+bita[2]*np.log(k[i-1]))
    c[i] = (bet*Pea[i])**(-1/sig)
    y[i] = shock[i]*(k[i-1]**alf)
    inv[i] = shock[i]*(k[i-1]**alf)-c[i]
    k[i] = inv[i]+(1-delta)*k[i-1]

# Deviations from steady state
c              = np.log(c/cs)
k              = np.log(k/ks)
y              = np.log(y/ys)
inv            = np.log(inv/ins)


LOG_DEV = np.concatenate((c,k,y,inv), axis=1)

# Plots the Impulse Response Functions
fig, han = plt.subplots()
han.plot(np.arange(1,NS), LOG_DEV[0:NS-1], label=['consumption', 'capital', \
                                                  'output', 'investment'], linewidth = 2)

han.set(title='Response to a one percent deviation in technology')
han.grid()
han.legend()

fig.savefig("growth_PEA_IRF.jpg", dpi=800)
plt.show()




# Calculate the statistics

while True:
    try:
        NR = int(input('Enter number of periods to simulate for statistics: '))
    except ValueError:
        print("Please enter a valid integer")
        continue
    else:
        if NR < 2:
            print("Please enter a positive integer greater than 1")
            continue
        else:
            print(f'You entered: {NR}')
            break

print(' ')

NR = NR+1

#sd = np.zeros((1,4))
#rd = np.zeros((1,4))

shock          = np.zeros((NR,1))
shock[0]       = 1
Pea            = np.zeros((NR,1))

c           = np.zeros((NR,1)); c[0] = cs
k           = np.zeros((NR,1)); k[0] = ks
y           = np.zeros((NR,1)); y[0] = ys
inv         = np.zeros((NR,1)); inv[0] = ins

#np.random.seed(1337)

for i in np.arange(1,NR):
    shock[i] = np.exp(rhoz*np.log(shock[i-1])+sigepsz*np.random.randn())
    Pea[i] = np.exp(bita[0]+bita[1]*np.log(shock[i])+bita[2]*np.log(k[i-1]))
    c[i] = (bet*Pea[i])**(-1/sig)
    y[i] = shock[i]*(k[i-1]**alf)
    inv[i] = shock[i]*(k[i-1]**alf)-c[i]
    k[i] = inv[i]+(1-delta)*k[i-1]


c           = np.log(c/cs)
k           = np.log(k/ks)
y           = np.log(y/ys)
inv         = np.log(inv/ins)

c       = hp1(c,1600)[0]
k       = hp1(k,1600)[0]
y       = hp1(y,1600)[0]
inv     = hp1(inv,1600)[0]

sd    = np.std(np.concatenate((c[1:NR],k[1:NR],y[1:NR],inv[1:NR]), axis=1),axis=0,ddof=1)
rd    = sd/sd[2]
print('  ')
print(' The average standard deviations for the three variables are:')
print('    c      k      y      in                                  ')
print(sd)
print('  ')
print(' The relative standard deviations for the three variables are:')
print('    c      k      y      in                                  ')
print(rd)





