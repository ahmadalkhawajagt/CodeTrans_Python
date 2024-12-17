# -*- coding: utf-8 -*-
"""

Testing the functions from Flodén
"""
import numpy as np
# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals 
# instead of using scientific notation
np.set_printoptions(suppress=True,precision=3,linewidth=np.inf)

from Floden import addacooper, tauchen

N = 15
rho = 0.9500
sigma = np.sqrt(0.030)
Z,PI = addacooper(N,0,rho,sigma)
print("Z =\n", Z)
print("PI =\n", PI)


N = 15
rho = 0.9500
sigma = np.sqrt(0.030)
Z,PI = tauchen(N,0,rho,sigma,1.2*np.log(N))
print("Z =\n", Z)
print("PI =\n", PI)