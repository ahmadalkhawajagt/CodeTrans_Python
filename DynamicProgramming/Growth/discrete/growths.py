"""

This program solves the stochastic growth model with CRRA utility 
and Markovian shocks using value function iteration and a discrete state space.
Iterations performed with loops.

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np
import matplotlib.pyplot as plt


# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# you can either import all the functions defined in the Markov file
# from Markov import *

# or only import the specific functions you want to use
from Markov import markov_approx, markov_chain

# Model Parameters
delta = 0.1
alfa = 0.33
A = 1
beta = 0.9
gamma = 2


# Algorithm parameters
simyes = 1 # simulates model
stationary = 1 # finds stationary distribution for capital
tolv = 1e-7 # tolerance, equals 1*10^-7


# Define the type of shock
shock = 3
if shock == 1:
    # symmetric shock
    Pi = np.array([[0.8, 0.2],
        [0.2, 0.8]])
elif shock == 2:
    Pi = np.array([[0.40, 0.30, 0.20, 0.10],
        [0.25, 0.40, 0.25, 0.10],
        [0.10, 0.25, 0.40, 0.25],
        [0.10, 0.20, 0.30, 0.40]])
elif shock == 3:
    rho=0.95
    sigmae=0.00712
    N=7
    m=3
    Pi, teta, P, arho, asigma = markov_approx(rho,sigmae,m,N)
    teta=np.exp(teta)


P = np.linalg.matrix_power(Pi, 10000) # stationary distribution
#Pit = Pi.T

# Grid for shock
if shock==1:
    teta_min = 0.9
    teta_max = 1.1
    grteta = 0.2
    teta = np.arange(teta_min,teta_max,grteta)
elif shock==2:
    teta = np.array([1.0225, 1.01, 0.99, 0.9775])

lt=teta.size


# Steady state
Eteta = np.dot(P[0:1,:],teta)
k_ss = (A*Eteta*alfa*beta/(1-beta*(1-delta)))**(1/(1-alfa))
y_ss  = Eteta * A * (k_ss**alfa)
i_ss  = delta * k_ss
c_ss  = y_ss - i_ss


#print(Eteta)
#print(k_ss)
#print(y_ss)
#print(i_ss)
#print(c_ss)


# Grid for capital
lk = int(input('Enter the number of grid points for the capital: '))
k_min = 0.5*k_ss
k_max = 1.5*k_ss
grk = (k_max-k_min)/(lk-1)
k = np.arange(k_min,k_max+grk,grk)
#k = np.linspace(k_min,k_max,lk).T


#Construct the objective function
gk = lk*lk
c = np.zeros((gk,lt))
for t in np.arange(0,lt):
    for i in np.arange(0,lk):
        for j in np.arange(0,lk):
            c[i*lk+j,t] = A*teta[t]*k[i]**(alfa)+(1-delta)*k[i]-k[j]
            if c[i*lk+j,t]<0:
                c[i*lk+j,t] = 1e-07

if gamma==1:
    U = np.log(c)
else:
    U = (c**(1-gamma))/(1-gamma)



# Initialization of the value function
V0 = np.ones((lk,lt))
V1 = np.zeros((lk,lt))
print("norm = ", np.linalg.norm(V1-V0))
# Iterate on the value function
while np.linalg.norm(V1-V0) > tolv: 
    V0=V1.copy()
    for j in np.arange(0,lt):
        for i in np.arange(0,lk):
            V1[i,j]=(U[i*lk:(i+1)*lk,j]+np.dot(beta*V0,Pi.T[:,j])).max()
    print("norm = ", np.linalg.norm(V1-V0))



# Compute the optimal policy functions
optim = np.zeros((lk,lt)).astype(int)
for j in np.arange(0,lt):
    for i in np.arange(0,lk):
        optim[i,j] = (U[i*lk:(i+1)*lk,j]+np.dot(beta*V0,Pi.T[:,j])).argmax()

polk = k[optim]
polc = A*np.dot(teta[:,np.newaxis],np.ones((1,lk))).T\
    *(np.dot(k[:,np.newaxis]**(alfa),np.ones((1,lt))))+\
        (1-delta)*np.dot(k[:,np.newaxis],np.ones((1,lt)))-polk

#print(polk)
#print(polc)


# Find the stationary distribution
if lk < 10:
    stationary = 0


if stationary:
    i = 0
    while polk[i,0] >= k[i]:
        i = i + 1
        if i >= lk:
            break

    khigh=k[i-1]

    j=0;
    while polk[j,lt-1] >= k[j]:
        j = j + 1
        if j >= lk:
            break
    
    klow=k[j-1]
    
    print('The lower support of the stationary distribution is', np.around(khigh,4))
    print('The upper support of the stationary distribution is', np.around(klow,4))


# Make plots of value and policy functions
fig, ax = plt.subplots()
ax.plot(k, V1[:,0], 'b')
ax.plot(k, V1[:,-1], 'r--')

ax.set(xlabel='Capital Stock', title='Value Functions')
ax.grid()

fig.savefig("plot_value.jpg", dpi=800)
plt.show()



fig2, axs = plt.subplots(2, 1)

axs[0].plot(k, polk[:,0], 'b')
axs[0].plot(k, polk[:,-1], 'r--')
axs[0].plot(k, k, 'y--')
axs[0].set(xlabel='Capital Stock', title='Capital Policy Functions')
axs[0].grid()


axs[1].plot(k, polc[:,0], 'b')
axs[1].plot(k, polc[:,-1], 'r--')
axs[1].set(xlabel='Capital Stock', title='Consumption Policy Functions')
axs[1].grid()


plt.tight_layout()
plt.savefig('plot_policy.jpg', dpi=800)
plt.show()
plt.close(fig2)



# Simulation
if simyes == 1:
    T = int(input('Enter the number of periods for simulation: '))
    S = markov_chain(Pi,T,0,me=1) # Start in state 1
    indk = round(lk/2)-1 # Start with the first capital
    kopt = np.zeros((T+1,))
    shock = np.zeros((T,))
    output = np.zeros((T,))
    invest = np.zeros((T,))
    cons = np.zeros((T,))
    kopt[0] = polk[indk,0]
    #print(S)
    for i in np.arange(0,T):
        indk = optim[indk,S[0,i]]
        shock[i] = teta[S[0,i]]
        kopt[i+1]=k[indk]
        output[i] = A*teta[S[0,i]]*(kopt[i]**alfa)
        invest[i] = kopt[i+1]-(1-delta)*kopt[i]
        cons[i] = output[i]-invest[i]

        
    #serie = np.concatenate((chat, khat))
    #print(serie)
    #print('shock=\n',shock)
    #print('kopt=\n',kopt)
    #print('output=\n',output)
    #print('invest=\n',invest)
    #print('cons=\n',cons)
    
    # The hat-variables are percentage deviation from steady state
    khat  = (kopt - k_ss) / k_ss
    chat  = (cons - c_ss) / c_ss


    time = np.arange(0,T)
    
    
    # Make plots of value and policy functions
    fig3, ax3 = plt.subplots()
    ax3.plot(time, output, 'b',label='=Y')
    ax3.plot(time, invest, 'r',label='=I')
    ax3.plot(time, cons, 'g',label='=C')
    ax3.plot(time, kopt[1:T+1], 'y',label='=K')

    ax3.set(xlabel='Time', title='Simulation of all variables')
    ax3.grid()
    ax3.legend()

    fig3.savefig("plot_simul.jpg", dpi=800)
    plt.show()
    
    
    fig4, axs4 = plt.subplots(2, 1)

    axs4[0].plot(time, khat[0:T], 'b')
    axs4[0].axhline(y=0, color='r', linestyle='-')
    axs4[0].set(xlabel='Time', ylabel = 'Percentage deviation', \
                title='Stochastic simulation of capital around its steady state')
    axs4[0].grid()


    axs4[1].plot(time, chat, 'b')
    axs4[1].axhline(y=0, color='r', linestyle='-')
    axs4[1].set(xlabel='Time', ylabel = 'Percentage deviation', \
                title='Stochastic simulation of consumption around the steady state')
    axs4[1].grid()


    plt.tight_layout()
    plt.savefig('plot_policy.jpg', dpi=800)
    plt.show()
    plt.close(fig4)


    X = np.concatenate((output[:,np.newaxis], invest[:,np.newaxis], \
                            cons[:,np.newaxis], kopt[0:T][:,np.newaxis]),axis=1)
    #print(X)
    #print(X.shape)

    # Some summary statistics
    print('Correlation of output with investment:')
    print(np.corrcoef(output.T,invest.T))
    print('Correlation of output with consumption:')
    print(np.corrcoef(output.T,cons.T))
    print('Correlation of output with capital:')
    print(np.corrcoef(output.T,kopt[1:T+1].T))
    print('Volatility of output, investment, consumption, and capital')
    vol=np.std(X,axis=0)/np.mean(X,axis=0)
    print(vol)



