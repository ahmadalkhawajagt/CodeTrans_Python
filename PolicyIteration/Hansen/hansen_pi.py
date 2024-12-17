# Importing packages
import numpy as np

# The interp1d function is the equivalent of interp1 in Matlab
from scipy.interpolate import interp1d

# This is used for the root_scalar function
import scipy.optimize as opt

import matplotlib.pyplot as plt

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the acm() function
from acm import acm

# import the hp1() function
from hp1 import hp1

# import the mcsim() function
from mcsim import mcsim


# Declarations
beta = 0.99
sigma = 1
delta = 0.025
theta = 0.36
a = 2
Ez = 1   # expectation of shock
lam = 0.9


plots = 1
simu = 1
loadin = 0


# Shock
zz,pi = acm(0,0.95,0.00712,7,1)[0:2]
zz = np.exp(zz) #acm returns the log productivity shock

# Grid dimensions
ns = zz.size
N = 301
T = ns*N


# Grid on capital
k = np.zeros((N,1))
k[0] = 1
k[N-1] = 120
for i in np.arange(1,N-1):
    k[i]=k[0]+i*(k[N-1]-k[0])/(N-1)


# Define Steady state variables
xx = (1-beta*(1-delta))/(beta*theta*Ez)
yy = ((1/beta+delta-1)/theta*(1+(1-theta)/a)-delta)*a/((1-theta)*Ez)
l_ss = xx/yy  # for this set of parameters, the steady state labor approxiamately equals to 1/3
k_ss = xx**(1/(theta-1))*l_ss
y_ss = Ez*k_ss**theta*l_ss**(1-theta)
i_ss = delta*k_ss
c_ss = y_ss-i_ss


# Initialization of policies
r = np.zeros((N,ns))
w = np.zeros((N,ns))
c = np.zeros((N,ns))
h = np.zeros((N,ns))
kpr = np.zeros((N,ns))

if loadin==0:
    # Initial Policy Functions
    for m in np.arange(0,ns):
        for i in np.arange(0,N):
            h[i,m]=0.3
            r[i,m]=theta*zz[m]*h[i,m]**(1-theta)*k[i]**(theta-1)+1-delta
            w[i,m]=(1-theta)*zz[m]*h[i,m]**(-theta)*k[i]**(theta)
            c[i,m]=max(0.001,(zz[m]*h[i,m]**(1-theta)*k[i]**(theta)-delta*k[i]))
            kpr[i,m]=max(k[0],w[i,m]*h[i,m]+r[i,m]*k[i]-c[i,m])
            kpr[i,m]=min(kpr[i,m],k[N-1])

else:
    with open('polrbc.npy', 'rb') as f:
        polrbc = np.load(f)
    A = polrbc.copy()
    
    for i in np.arange(0,ns):
        for l in np.arange(0,N):
            c[l,i]=A[l+i*N,1]
            kpr[l,i]=A[l+i*N,2]
            h[l,i]=A[l+i*N,3]
            r[l,i]=A[l+i*N,4]
            w[l,i]=A[l+i*N,5]



cpp = np.zeros((ns,ns))
hp = np.zeros((ns,ns))

cn = np.zeros((N,ns))
hn = np.zeros((N,ns))
kprn = np.zeros((N,ns))

niter = 0
err = np.array([10.,10.,10.])
interma = 0

if loadin==0:
    tol = 0.0001
else:
    tol = 0.001

while np.amax(err)>tol:
    #if niter>1000:
    #    break
    
    niter=niter+1
    for m in np.arange(0,ns):
        for i in np.arange(0,N):
            if interma==1:
                cpp=(interp1d(k.squeeze(),c,axis=0)(kpr[i,m]))**(-sigma)
                hp=interp1d(k.squeeze(),h,axis=0)(kpr[i,m])
            else:
                mm = np.abs(kpr[i,m]-k).argmin()
                
                if (kpr[i,m]<=k[mm] and mm>0):
                    weight=(k[mm]-kpr[i,m])/(k[mm]-k[mm-1])
                    cpp=((weight*c[mm-1,:]+(1-weight)*c[mm,:])**(-sigma)).T
                    hp=(weight*h[mm-1,:]+(1-weight)*h[mm,:]).T;
                else:
                    weight=(k[mm+1]-kpr[i,m])/(k[mm+1]-k[mm])
                    cpp=((weight*c[mm,:]+(1-weight)*c[mm+1,:])**(-sigma)).T
                    hp=(weight*h[mm,:]+(1-weight)*h[mm+1,:]).T
                    

            cn[i,m]=max(0.001,(beta*np.sum(cpp*(theta*zz.squeeze()*hp.squeeze()**(1-theta)*\
                                                kpr[i,m]**(theta-1)+\
                                                1-delta)*(pi[m,:].T)))**(-1/sigma))

            func = lambda x: a*cn[i,m]-(1-theta)*zz[m]*(k[i]**theta)*(x**(-theta))*(1-x)
            hn[i,m] = opt.root_scalar(func, bracket=[0.00001,1-0.00001]).root
            if hn[i,m] < 0:
                hn[i,m] = 0
            
            if hn[i,m] > 1:
                hn[i,m] = 1
            
            
            r[i,m]=theta*zz[m]*hn[i,m]**(1-theta)*k[i]**(theta-1)+1-delta
            w[i,m]=(1-theta)*zz[m]*hn[i,m]**(-theta)*k[i]**theta
            kprn[i,m]=max(k[0],w[i,m]*hn[i,m]+r[i,m]*k[i]-cn[i,m])
            kprn[i,m]=min(kprn[i,m],k[N-1])

    err[0]=np.abs((c-cn)).max()
    err[1]=np.abs((kpr-kprn)).max()
    err[2]=np.abs((h-hn)).max()

    print("iteration =", niter, ", Error =", err[0], err[1], err[2])
    c=lam*c+(1-lam)*cn
    kpr=lam*kpr+(1-lam)*kprn
    h=lam*h+(1-lam)*hn


polrbc=np.zeros((N*ns,6))
for m in np.arange(0,ns):
    for i in np.arange(0,N):
        polrbc[i+m*N,0]=k[i]
        polrbc[i+m*N,1]=c[i,m]
        polrbc[i+m*N,2]=kpr[i,m]
        polrbc[i+m*N,3]=h[i,m]
        polrbc[i+m*N,4]=r[i,m]
        polrbc[i+m*N,5]=w[i,m]


with open('polrbc.npy', 'wb') as f:
    np.save(f, polrbc)

if plots:
    ccm = np.zeros((N,ns))
    hcm = np.zeros((N,ns))
    kprcm = np.zeros((N,ns))
    
    PC = polrbc.copy()
    KC = PC[0:N,0]
    
    for i in np.arange(0,ns):
        for l in np.arange(0,N):
            ccm[l,i]=PC[l+i*N,1]
            kprcm[l,i]=PC[l+i*N,2]
            hcm[l,i]=PC[l+i*N,3]

            
    # Plot capital, consumption, and labor policy functions
    fig1, ax1 = plt.subplots()
    
    ax1.plot(KC, kprcm)
    ax1.set(xlabel='Capital Stock', title='Capital Policy Function')
    ax1.grid()
    
    plt.tight_layout()
    plt.savefig('Hansen_PI_capital_policy.jpg', dpi=800)
    plt.show()
    plt.close(fig1)

    fig2, ax2 = plt.subplots()
    
    ax2.plot(KC, ccm)
    ax2.set(xlabel='Capital Stock', title='Consumption Policy Function')
    ax2.grid()
    
    plt.tight_layout()
    plt.savefig('Hansen_PI_consumption_policy.jpg', dpi=800)
    plt.show()
    plt.close(fig2)
    

    fig3, ax3 = plt.subplots()
    
    ax3.plot(KC, hcm)
    ax3.set(xlabel='Capital Stock', title='Labor Policy Function')
    ax3.grid()
    
    plt.tight_layout()
    plt.savefig('Hansen_PI_labor_policy.jpg', dpi=800)
    plt.show()
    plt.close(fig3)
    

if simu:
    np.random.seed(1337)
    T = 115
    N = 100
    
    std_mat = np.zeros((N,6))
    cc_mat = np.zeros((N,6))
    
    for j in np.arange(0,N):
        S = mcsim(zz,pi,T)
        sz = S[0]
        
        kt = np.zeros((T+1,))
        it = np.zeros((T,))
        ct = np.zeros((T,))
        yt = np.zeros((T,))
        lt = np.zeros((T,))
        prot = np.zeros((T,))
        zt = np.zeros((T,))
        
        kt[0] = k_ss
        
        for i in np.arange(0,T):
            sz = S[i]
            zt[i] = zz[sz]
            
            zzz = np.abs((k-kt[i])).argmin()
            
            mm = zzz
            if (kt[i]<=k[mm] and mm>0):
                weight=(k[mm]-kt[i])/(k[mm]-k[mm-1])
                kt[i+1]=weight*kpr[mm-1,sz]+(1-weight)*kpr[mm,sz]
                lt[i]=weight*h[mm-1,sz]+(1-weight)*h[mm,sz]

            else:
                weight=(k[mm+1]-kt[i])/(k[mm+1]-k[mm])
                kt[i+1]=weight*kpr[mm,sz]+(1-weight)*kpr[mm+1,sz]
                lt[i]=weight*h[mm,sz]+(1-weight)*h[mm+1,sz]

            
                        
            yt[i]=zt[i]*lt[i]**(1-theta)*kt[i]**theta;
            ct[i]=(yt[i]+(1-delta)*kt[i]-kt[i+1])
            it[i]=kt[i+1]-(1-delta)*kt[i]
            prot[i]=yt[i]/lt[i]
        
        
        kk = np.log(kt[0:T])[np.newaxis,:]
        yy = np.log(yt[0:T])[np.newaxis,:]
        cc = np.log(ct[0:T])[np.newaxis,:]
        inn = np.log(it[0:T])[np.newaxis,:]
        hh = np.log(lt[0:T])[np.newaxis,:]
        prodd = np.log(prot[0:T])[np.newaxis,:]
        
        
        dhp, dtr = hp1(np.concatenate((yy.T, inn.T, cc.T, kk.T, hh.T, prodd.T),axis=1), 1600)
        std_mat[j,:] = np.std(dhp,axis=0,ddof=1)*100
        Corr = np.corrcoef(dhp,rowvar=False)
        cc_mat[j,:] = Corr[:,0]
        
        
        
    std = np.mean(std_mat,axis=0)
    corr = np.mean(cc_mat,axis=0)
    
    
    print('HANSEN: std(x)/std(y) corr(x,y) for y, i, c, k, h, prod')
    print(np.concatenate((np.array([[1.36, 4.24, 0.42, 0.36, 0.7, 0.68]]).T/1.36, \
                          np.array([[1, 0.99, 0.89, 0.06, 0.98, 0.98]]).T),axis=1))
    print('std(x) std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
    print(np.concatenate((std[:,np.newaxis], (std/std[0])[:,np.newaxis], corr[:,np.newaxis]), \
                         axis=1))



