# -*- coding: utf-8 -*-
"""

Testing the functions from Sargent
"""
import numpy as np
# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals 
# instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

from Sargent import approx_markov

rho=0.95
sigmae=0.01
N=7
m=3
x, P = approx_markov(rho,sigmae,m,N)
print("Transition Matrix:\n", P)
print("Discretized State Space:\n", x)