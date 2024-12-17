"""
The Hodrick-Prescott filter function
"""

# Importing packages
import numpy as np

def hp(T, lamb):
    matrix = np.zeros((T, T))

    matrix[0, 0:3] = [1 + lamb, -2 * lamb, lamb]
    matrix[1, 0:4] = [-2 * lamb, 1 + 5 * lamb, -4 * lamb, lamb]

    for i in np.arange(3,T-1):
        matrix[i-1, i-3 : i+2] = [lamb, -4*lamb, 1 + 6 * lamb, -4 * lamb, lamb]

    matrix[T-2, T-4:T] = [lamb, -4 * lamb, 1 + 5 * lamb, -2 * lamb]
    matrix[T-1, T-3:T] = [lamb, -2 * lamb, 1 + lamb]

    return matrix