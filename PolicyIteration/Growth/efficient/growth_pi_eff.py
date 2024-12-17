"""
Solving Brock-Mirman optimal growth model with PI

Translated from Eva Carceles-Poveda's MATLAB codes
"""


# Importing packages
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)


# this is used to save the variables of the current session
import shelve

# import the acm() function
from acm import acm

# import the growthvipi() function
from growthvipi import growthvipi



alpha = 0.36
beta = 0.99
gamma = 1       # log utility
delta = 0.025
PAR = np.array([alpha,beta,delta,gamma])



bk = (1 / beta / alpha + (delta - 1) / alpha)**(1 / (alpha - 1))
bi = delta * bk
bc = bk**alpha - bi
DSS = np.array([bk,bc,bi])


SSS, TM = acm(0,0.95,0.00712,7,1)[0:2]
SSS = np.exp(SSS)     # acm returns the log productivity shock


krange = np.array([0.9,1.1])
nk = 200
SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk)  #kgrid



print('Method for solving the model:\n')
print('-- 1 policy iteration with interpolation\n')
print('     and discrete maximization;\n')
print('-- 2 value function iteration with interpolation\n')
print('     and discrete maximization;\n')
print('-- 3 value function iteration with interpolation\n')
print('     and continuous maximization.\n')

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


#--------------------------------------------------------------------------
# First, find the ergodic set for chosen method
#--------------------------------------------------------------------------

while True:
    # Request Unprocessed Text Input, y = yes, n = no
    findergk = input('Find the ergodic set first (y/n): ')

    if findergk not in ['y','n']:
        print('Please enter a valid response [y/n]\n')
        continue
    else:
        print(f'You entered {findergk}')
        break



if findergk == 'y':
    match me:
        case 1:
            v,gk,gc,gi,gy,t = growthvipi(0,0,PAR, DSS, SSK, SSS, TM)
            
            ergk_pi = SSK[[np.where(gk[:,0] > SSK)[0][-1]+1,np.where(gk[:,-1] < SSK)[0][0]-1]]/bk
            print('ergodic set of capital:', ergk_pi)
            with open('ergkpi.npy', 'wb') as f:
                np.save(f, ergk_pi)
            
        case 2:
            v,gk,gc,gi,gy,t = growthvipi(1,0,PAR, DSS, SSK, SSS, TM)
            
            ergk_vi_dsc = SSK[[np.where(gk[:,0] > SSK)[0][-1]+1,np.where(gk[:,-1] < SSK)[0][0]-1]]/bk
            print('ergodic set of capital:', ergk_vi_dsc)
            with open('ergkvid.npy', 'wb') as f:
                np.save(f, ergk_vi_dsc)
            
        case 3:
            v,gk,gc,gi,gy,t = growthvipi(1,1,PAR, DSS, SSK, SSS, TM)
            
            ergk_vi_ctn = SSK[[np.where(gk[:,0] > SSK)[0][-1]+1,np.where(gk[:,-1] < SSK)[0][0]-1]]/bk
            print('ergodic set of capital:', ergk_vi_ctn)
            with open('ergkvic.npy', 'wb') as f:
                np.save(f, ergk_vi_ctn)

            

#--------------------------------------------------------------------------
# Solve the model over ergodic set
#--------------------------------------------------------------------------

while True:
    # Request Unprocessed Text Input, y = yes, n = no
    solvemodel = input('Solve the model using existing ergk (y/n): ')

    if solvemodel not in ['y','n']:
        print('Please enter a valid response [y/n]\n')
        continue
    else:
        print(f'You entered {solvemodel}')
        break


if solvemodel == 'y':
    nk=100
    match me:
        case 1:
            with open('ergkpi.npy', 'rb') as f:
                ergk_pi = np.load(f)

            krange = ergk_pi
            SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk) 

            v,gk,gc,gi,gy,t = growthvipi(0,0,PAR, DSS, SSK, SSS, TM)
            
            save_shelf = shelve.open('./bm_pi_ergk.pkl','n')
            for k in dir():
                try:
                    save_shelf[k] = globals()[k]
                except Exception:
                    pass
            save_shelf.close()
            
        case 2:
            with open('ergkvid.npy', 'rb') as f:
                ergk_vi_dsc = np.load(f)
            
            krange = ergk_vi_dsc
            SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk) 
            
            v,gk,gc,gi,gy,t = growthvipi(1,0,PAR, DSS, SSK, SSS, TM)
            
            save_shelf = shelve.open('./bm_vi_dsc_ergk.pkl','n')
            for k in dir():
                try:
                    save_shelf[k] = globals()[k]
                except Exception:
                    pass
            save_shelf.close()
            
        case 3:
            with open('ergkvic.npy', 'rb') as f:
                ergk_vi_ctn = np.load(f)
            
            krange = ergk_vi_ctn
            SSK = np.linspace(krange[0] * bk, krange[1] * bk, nk) 
            
            v,gk,gc,gi,gy,t = growthvipi(1,1,PAR, DSS, SSK, SSS, TM)
            
            save_shelf = shelve.open('./bm_vi_ctn_ergk.pkl','n')
            for k in dir():
                try:
                    save_shelf[k] = globals()[k]
                except Exception:
                    pass
            save_shelf.close()


    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = "14"

    # create objects
    fig1 = plt.figure(constrained_layout=True)
    gs = GridSpec(3, 2, figure=fig1)

    # create sub plots as grid
    ax1 = fig1.add_subplot(gs[0, 0])
    ax2 = fig1.add_subplot(gs[0, 1])
    ax3 = fig1.add_subplot(gs[1, 0])
    ax4 = fig1.add_subplot(gs[1, 1])
    ax5 = fig1.add_subplot(gs[2, :])

    ax1.plot(SSK, gk[:,1], 'k-')
    ax1.plot(SSK, gk[:,-1], 'k--')
    ax1.set(title='$g_{k}(k)$')

    ax2.plot(SSK, gi[:,1], 'k-')
    ax2.plot(SSK, gi[:,-1], 'k--')
    ax2.set(title='$g_{i}(k)$')

    ax3.plot(SSK, gc[:,1], 'k-')
    ax3.plot(SSK, gc[:,-1], 'k--')
    ax3.set(title='$g_{c}(k)$')

    ax4.plot(SSK, gy[:,1], 'k-')
    ax4.plot(SSK, gy[:,-1], 'k--')
    ax4.set(title='$g_{y}(k)$')

    ax5.plot(SSK, v[:,1], 'k-')
    ax5.plot(SSK, v[:,-1], 'k--')
    ax5.set(xlabel = '$k$ over the ergodic set', title='$V(k)$')


    plt.tight_layout()
    plt.savefig('growth_pi_eff_value_policy.jpg', dpi=800)
    plt.show()
    plt.close(fig1)



#for restoring variables

#import shelve

#bk_restore = shelve.open('./your_bk_shelve.pkl')
#for k in bk_restore:
#    globals()[k] = bk_restore[k]
#bk_restore.close()




