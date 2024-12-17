"""
Solves Hansen model with continuos DP and efficient algorithm

Translated from Eva Carceles-Poveda's MATLAB codes
"""

# Importing packages
import numpy as np

import matplotlib.pyplot as plt

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# import the acm() function
from acm import acm

# import the hp() function
from hp import hp


# import the rbcvipi() and mcsim() functions
from rbcvipi import rbcvipi,linitp
from mcsim import mcsim

import shelve


alpha = 0.36
beta = 0.99
gamma = 1       # log utility
delta = 0.025
a = 2/3        
PAR = np.array([alpha,beta,delta,gamma,a])  # a governs labor supply


# Discretize shock
SSS, TM = acm(0,0.95,0.00712,7,1)[0:2]
SSS = np.exp(SSS)


# Steady state
br = (1 / alpha / beta + (delta - 1) / alpha)**(1 / (1 - alpha))
bk = (1 - a) * (1 - alpha) / ((1 - alpha + a * alpha) * br - a * delta * br**alpha)
bh = br * bk
# del br
bi = delta * bk
bc = bk**alpha * bh**(1 - alpha) - bi
DSS = np.array([bk,bc,bh,bi])


# Discretize capital
krange = np.array([0.85,1.15])
nk = 50
SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk)


print('Choose the method for solving the model:\n')
print('-- 1 policy iteration with interpolation\n')
print('-- 2 discrete value function iteration with interpolation\n')
print('-- 3 continuous value function iteration with interpolation\n')


while True:
    try:
        me = int(input('-- '))
    except ValueError:
        print("Please enter a valid integer [1-2-3]")
        continue
    else:
        if me not in [1,2,3]:
            print("Please enter a valid integer [1-2-3]")
            continue
        else:
            print(f'You entered: {me}')
            break


while True:
    # Request Unprocessed Text Input, y = yes, n = no
    findergk = input('Find the ergodic set first (y/n): ') 

    if findergk not in ['y','n']:
        print('Please enter a valid response [y/n]\n')
        continue
    else:
        print(f'You entered {findergk}')
        break


#--------------------------------------------------------------------------
# First, find the ergodic set for chosen method
#--------------------------------------------------------------------------

if findergk == 'y':
    match me:
        case 1:
            v,gk,gh,gc,gi,gy,t = rbcvipi(0,0,PAR, DSS, SSK, SSS, TM)
            
            ergk_pi = SSK[[np.where(gk[:,0] > SSK)[0][-1]+1,\
                           np.where(gk[:,-1] < SSK)[0][0]-1]]/bk
            print('ergodic set of capital:', ergk_pi)
            with open('ergk.npy', 'wb') as f:
                np.save(f, ergk_pi)
            
        case 2:
            v,gk,gh,gc,gi,gy,t = rbcvipi(1,0,PAR, DSS, SSK, SSS, TM)
            
            ergk_vi_dsc = SSK[[np.where(np.around(gk[:,0],4) > np.around(SSK,4))[0][-1]+1,\
                               np.where(np.around(gk[:,-1],4) < np.around(SSK,4))[0][0]-1]]/bk
            print('ergodic set of capital:', ergk_vi_dsc)
            with open('ergk.npy', 'wb') as f:
                np.save(f, ergk_vi_dsc)            
            
        case 3:
            v,gk,gh,gc,gi,gy,t = rbcvipi(1,1,PAR, DSS, SSK, SSS, TM)
            
            ergk_vi_ctn = SSK[[np.where(gk[:,0] > SSK)[0][-1]+1,\
                               np.where(gk[:,-1] < SSK)[0][0]-1]]/bk
            print('ergodic set of capital:', ergk_vi_ctn)
            with open('ergk.npy', 'wb') as f:
                np.save(f, ergk_vi_ctn)
            

      

while True:
    # Request Unprocessed Text Input, y = yes, n = no
    solvemodel = input('Solve the model using existing ergk (y/n): ')

    if solvemodel not in ['y','n']:
        print('Please enter a valid response [y/n]\n')
        continue
    else:
        print(f'You entered {solvemodel}')
        break



#--------------------------------------------------------------------------
# Solve the model over ergodic set
#--------------------------------------------------------------------------
if solvemodel == 'y':
    nk=50
    match me:
        case 1:
            ergk_pi = np.load('./ergk.npy')

            krange = ergk_pi
            SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk) 

            v,gk,gh,gc,gi,gy,t = rbcvipi(0,0,PAR, DSS, SSK, SSS, TM)
            
            save_shelf = shelve.open('./bm_pi_ergk.pkl','n')
            for k in dir():
                try:
                    save_shelf[k] = globals()[k]
                except Exception:
                    pass
            save_shelf.close()
            
        case 2:
            ergk_vi_dsc = np.load('./ergk.npy')
            
            krange = ergk_vi_dsc
            SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk) 
            
            v,gk,gh,gc,gi,gy,t = rbcvipi(1,0,PAR, DSS, SSK, SSS, TM)
            
            save_shelf = shelve.open('./bm_vi_dsc_ergk.pkl','n')
            for k in dir():
                try:
                    save_shelf[k] = globals()[k]
                except Exception:
                    pass
            save_shelf.close()
            
        case 3:    
            ergk_vi_ctn = np.load('./ergk.npy')
            
            krange = ergk_vi_ctn
            SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk) 
            
            v,gk,gh,gc,gi,gy,t = rbcvipi(1,1,PAR, DSS, SSK, SSS, TM)
            
            save_shelf = shelve.open('./bm_vi_ctn_ergk.pkl','n')
            for k in dir():
                try:
                    save_shelf[k] = globals()[k]
                except Exception:
                    pass
            save_shelf.close()




    fig1, axs1 = plt.subplots(3, 2)

    axs1[0,0].plot(SSK, gk[:,0], 'k-')
    axs1[0,0].plot(SSK, gk[:,-1], 'k--')
    axs1[0,0].set(xlabel='$k$', title='$g_{k}(k)$')

    axs1[0,1].plot(SSK, gh[:,0], 'k-')
    axs1[0,1].plot(SSK, gh[:,-1], 'k--')
    axs1[0,1].set(xlabel='$k$', title='$g_{h}(k)$')

    axs1[1,0].plot(SSK, gi[:,0], 'k-')
    axs1[1,0].plot(SSK, gi[:,-1], 'k--')
    axs1[1,0].set(xlabel='$k$', title='$g_{i}(k)$')

    axs1[1,1].plot(SSK, gc[:,0], 'k-')
    axs1[1,1].plot(SSK, gc[:,-1], 'k--')
    axs1[1,1].set(xlabel='$k$', title='$g_{c}(k)$')

    axs1[2,0].plot(SSK, gy[:,0], 'k-')
    axs1[2,0].plot(SSK, gy[:,-1], 'k--')
    axs1[2,0].set(xlabel='$k$', title='$g_{y}(k)$')

    axs1[2,1].plot(SSK, v[:,0], 'k-')
    axs1[2,1].plot(SSK, v[:,-1], 'k--')
    axs1[2,1].set(xlabel='$k$', title='$V(k)$')

    plt.tight_layout()
    plt.savefig('Hansen_CDP_eff_policy_value.jpg', dpi=800)
    plt.show()
    plt.close(fig1)


    #--------------------------------------------------------------------------
    # Simulation for Hansen model
    #--------------------------------------------------------------------------
    # np.random.seed(1337)

    ss = SSS
    tm = TM
    ns = ss.size
    ssk = SSK
    nk = ssk.size

    T = 115
    N = 100

    w = 1600   # Weight of HP filter
    fm = hp(T,w)   # get the filtering matrix fm

    stat = np.zeros((N,12))     # store statistics for each simulation
    hpd = np.zeros((T,6))

    k = np.ones((T,1))
    ind = np.floor(nk / 2).astype(int)-1
    kstep = (ssk[-1] - ssk[0]) / (nk - 1)
    ssk_grid = np.array([ssk[0],(ssk[-1] - ssk[0]) / (nk - 1)])

    for n in np.arange(0,N):
        Is = mcsim(ss,tm,T)
        
        k[0] = ssk[ind]
        for t in np.arange(1,T):
            j = 1
            k[t] = linitp(k[t-1],ssk_grid,gk[:,Is[t-1]])

        logsimk = np.log(k)
        logsimi = np.log(linitp(k.T,ssk_grid,gi[:,Is.squeeze()]).T)
        logsimc = np.log(linitp(k.T,ssk_grid,gc[:,Is.squeeze()]).T)
        logsimh = np.log(linitp(k.T,ssk_grid,gh[:,Is.squeeze()]).T)
        logsimy = np.log(linitp(k.T,ssk_grid,gy[:,Is.squeeze()]).T)
        logsimp = logsimy - logsimh  # productivity
        
        hpd[:,[0]] = logsimy - np.linalg.solve(fm,logsimy)
        hpd[:,[1]] = logsimc - np.linalg.solve(fm,logsimc)
        hpd[:,[2]] = logsimi - np.linalg.solve(fm,logsimi)
        hpd[:,[3]] = logsimk - np.linalg.solve(fm,logsimk)
        hpd[:,[4]] = logsimh - np.linalg.solve(fm,logsimh)
        hpd[:,[5]] = logsimp - np.linalg.solve(fm,logsimp)
        
        stat[n,0:6] = np.std(hpd,axis=0)
        stat[n,6:12] = np.corrcoef(hpd[:,0],hpd,rowvar=False)[0,1:]
        
    A=np.mean(stat,axis=0)
    print('HANSEN: std(x)/std(y) corr(x,y) for y, c, i, k, h, prod')
    print(np.concatenate((np.array([[1.36, 0.42, 4.24, 0.36, 0.7, 0.68]]).T/1.36, \
                      np.array([[1, 0.89, 0.99, 0.06, 0.98, 0.98]]).T),axis=1))
    print('std(x) std(x)/std(y) corr(x,y) for y, i, c, k, h, prod:')
    print(np.concatenate(((A[0:6]*100)[:,np.newaxis], \
                          ((A[0:6]*100)/(A[0]*100))[:,np.newaxis], A[6:12][:,np.newaxis]),axis=1))





#for restoring variables

#import shelve

#bk_restore = shelve.open('./your_bk_shelve.pkl')
#for k in bk_restore:
#    globals()[k] = bk_restore[k]
#bk_restore.close()