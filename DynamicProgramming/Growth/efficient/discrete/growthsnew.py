"""
Dynamic programming, efficient, discrete

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np
import matplotlib.pyplot as plt

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the dpnew() function
from dpnew import dpnew

# Parameters
alpha = 0.33
beta = 0.9
sigma = 2
delta = 0.1
PAR = np.array([alpha,beta,sigma,delta])

#print(PAR)


# Shock
SSS = np.array([0.9,1.1])
TM = np.array([[0.8, 0.2],[0.2, 0.8]])

#print(SSS)
#print(TM)


# Steady State
kb = (1 / beta / alpha + (delta - 1) / alpha)**(1 / (alpha - 1))
ib = delta * kb
cb = kb**alpha - ib

DSS = np.array([kb,cb,ib])

#print(DSS)


# Capital grid
NK = 200
kmax = 1.5*kb
nk = NK
kminmax = np.array([0.5*kb,kmax]) # kminmax is a two elements vector with the min and max of capital
Kgrid = np.linspace(kminmax[0],kminmax[1],nk)

#print(Kgrid)

v, gk = dpnew(0, Kgrid, PAR, DSS, SSS, TM)
gc=np.dot(SSS[:,np.newaxis],np.ones((1,nk))).T*(np.dot((Kgrid[:,np.newaxis]**alpha),np.ones((1,SSS.size))))+(1-delta)*Kgrid[:,np.newaxis]*np.ones((1,SSS.size))-Kgrid[gk]


#print(v)
#print(gk)
#print(gc)


# Make plots of value and policy functions
fig, ax = plt.subplots()
ax.plot(Kgrid, v[:,0], 'b')
ax.plot(Kgrid, v[:,-1], 'r--')

ax.set(xlabel='Capital Stock', title='Value Functions')
ax.grid()

fig.savefig("plot_value.jpg", dpi=800)
plt.show()


fig2, axs = plt.subplots(2, 1)

axs[0].plot(Kgrid, Kgrid[gk[:,0]], 'b')
axs[0].plot(Kgrid, Kgrid[gk[:,-1]], 'r--')
axs[0].plot(Kgrid, Kgrid, 'y--')
axs[0].set(xlabel='Capital Stock', title='Capital Policy Functions')
axs[0].grid()


axs[1].plot(Kgrid, gc[:,0], 'b')
axs[1].plot(Kgrid, gc[:,-1], 'r--')
axs[1].set(xlabel='Capital Stock', title='Consumption Policy Functions')
axs[1].grid()


plt.tight_layout()
plt.savefig('plot_policy.jpg', dpi=800)
plt.show()
plt.close(fig2)



