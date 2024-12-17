"""
The riccati1 (Ricatti difference equation) function
"""

# Importing packages
import numpy as np

def riccati1(U,DJ,DH,S,C,B,Sigma,beta):
    """ Syntax: [P1,J,d] = riccati(U,DJ,DH,S,C,B,Sigma,beta)
    
    This function solves for the value and policy functions of a linear quadratic problem 
    by iterating on the Ricatti difference equation. The inputs are the return function at 
    steady state (U), the Jacobian and Hessian matrices (DJ and DH), the vector of states 
    (S=[z; s]), the vector of controls (C=[x]), the matrix B satisfying [1;z';s']=B[1;z;s;x] 
    the variance covariance matrix of [1 z' s'] (Sigma) and the discount factor beta.
    
    Translated from Eva Carceles-Poveda 2003 MATLAB code
    """

    tolv = 1e-07

    ns1, ns2 = S.shape
    if ns2 > ns1:
        S = S.T

    ns = max(S.shape)

    nc1, nc2 = C.shape
    if nc2 > nc1:
        C = C.T
    
    nc = max(C.shape)

    WW = np.concatenate((S.T,C.T),axis=1).T
    Q11 = U - WW.T@DJ + 0.5*WW.T@DH@WW
    Q12 = 0.5*(DJ-DH@WW)
    Q22 = 0.5*DH
        
    QQ = np.concatenate((np.concatenate((Q11, Q12.T),axis=1),np.concatenate((Q12,Q22),axis=1)))

    nq=ns+nc+1

    # Partition Q to separate states and controls 
    Qff = QQ[0:ns+1,0:ns+1]
    Qfx = QQ[ns+1:nq,0:ns+1]
    Qxx = QQ[nq-nc:nq,nq-nc:nq]
    
    # Initialize matrices
    P0 = -0.1*np.eye((ns+1))
    P1 = np.ones((ns+1,ns+1))
    
    # Iterate on Bellman's equation until convergence
    while np.linalg.norm(np.abs(P1-P0)) > tolv:
        P1 = P0.copy()
        M = B.T@P0@B;
        Mff = M[0:ns+1,0:ns+1]
        Mfx = M[ns+1:nq,0:ns+1]
        Mxx = M[nq-nc:nq,nq-nc:nq]

        P0=Qff+beta*Mff-(Qfx+beta*Mfx).T@np.linalg.inv(Qxx+beta*Mxx)@(Qfx+beta*Mfx)   

      

    J = -np.linalg.inv(Qxx + beta*Mxx)@(Qfx + beta*Mfx)

    d = beta/(1-beta)*np.trace(P0@Sigma)

    return P1,J,d