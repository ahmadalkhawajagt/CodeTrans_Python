# -*- coding: utf-8 -*-
"""

Testing the functions from the Markov file
"""
# importing packages
import numpy as np

# needed for compact printing of numpy arrays
# use precision to set the number of decimal digits to display
# use suppress=True to show values in full decimals 
# instead of using scientific notation
np.set_printoptions(suppress=True,precision=4,linewidth=np.inf)

# you can either import all the functions defined in the Markov file
# from Markov import *

# or only import the specific functions you want to use
from Markov import ln_shock, ln_shock_m, markov_approx, markov_chain, \
    ergodic, iid


# Testing ln_shock
print(ln_shock(0.1, 0.001, 100))

# Testing ln_shock_m
print(ln_shock_m(-0.0631, 0.9, 0.05, 100))


# Testing markov_chain
transition_matrix = np.array([[1, 2], [2, 5]])
print(markov_chain(transition_matrix))

transition_matrix2 = np.array([[1, 2, 3], [2, 5, 6]])
print(markov_chain(transition_matrix2))

transition_matrix3 = np.array([[1, 4], [2, 5]])
print(markov_chain(transition_matrix3, 100, 3, 2))

pi = np.array([[0.8010, 0.1837, 0.0147, 0.0005, 0.0000, 0.0000, 0.0000],
    [0.1831, 0.5308, 0.2419, 0.0414, 0.0027, 0.0000, 0.0000],
    [0.0148, 0.2417, 0.4401, 0.2495, 0.0512, 0.0026, 0.0000],
    [0.0005, 0.0414, 0.2495, 0.4171, 0.2495, 0.0414, 0.0005],
    [0.0000, 0.0026, 0.0512, 0.2495, 0.4401, 0.2417, 0.0148],
    [0.0000, 0.0000, 0.0027, 0.0414, 0.2419, 0.5308, 0.1831],
    [0.0000, 0.0000, 0.0000, 0.0005, 0.0147, 0.1837, 0.8010]])

markov_chain(pi, 115, 1, 2)



# Testing ergodic
trans=np.array([[0.8, 0.1, 0.1], [0, 0.2, 0.8], [0.7, 0.3, 0]])
va, vec = np.linalg.eig(trans.T)
va = np.diag(va)
r1, r2 = (np.absolute(va-1)<1e-014).nonzero()
print("r1 =", r1, "r2 =", r2)
print(r1.size)
print(np.absolute(va-1)<1e-014)
print(vec)
print(va)


trans=np.array([[0.8, 0.1, 0.1], [0, 0.2, 0.8], [0.7, 0.3, 0]])
print("Method 1:")
print("Stationary Distribution:\n", ergodic(trans,1))
print("Method 2:")
print("Stationary Distribution:\n", ergodic(trans,2))
print("Method 3:")
print("Stationary Distribution:\n", ergodic(trans,3))



# Testing iid
tt = ergodic(trans)
print("Method 1:")
mm=iid(tt,100,1)
print((mm == 1).sum())
print((mm == 2).sum())
print((mm == 3).sum())

print("Method 2:")
mm=iid(tt,100,2)
print((mm == 1).sum())
print((mm == 2).sum())
print((mm == 3).sum())



# Testing markov_approx
rho=0.95
sigmae=0.01
N=7
m=3
Pi, teta, P, arho, asigma = markov_approx(rho,sigmae,m,N)
print("Transition Matrix:\n", Pi)
print("Discretized State Space:\n", teta)
print("The chain stationary distribution:\n", P)
print("Theoretical first order autoregression coefficient:\n", arho)
print("Theoretical standard deviation:\n", asigma)

print(markov_approx.__doc__)