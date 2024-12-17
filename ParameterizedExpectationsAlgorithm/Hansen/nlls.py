# Importing packages
import numpy as np

# Non-linear least squares
def nlls(DEP,IND1,IND2,T,IBETA):
    
    GBETA = np.zeros((3,1))

    HALT = 0

    DER1 = np.zeros((T-1,1))
    DER2 = np.zeros((T-1,1))
    DER3 = np.zeros((T-1,1))
    F = np.zeros((T-1,1))

    MAX_IT = 100
    smooth = 0.6
    prec = 0.001

    it = 0

    while (HALT == 0 and it < MAX_IT):

        # TAYLOR AROUND IBETA

        IBETA = (smooth*IBETA) + (1-smooth)*GBETA


        DER1 = np.exp(IBETA[0]*np.ones((T-1,1))+(IBETA[1]*IND1)+(IBETA[2]*IND2))
        F = DER1.copy()
        DER2 = F * IND1
        DER3 = F * IND2

        DER = np.concatenate((DER1,DER2,DER3), axis=1)

        # 1.1 Estimate coefficients of linearized equation
        #################################################################
        # DEP - F + DER(Tx3)*IBETA(3x1) = DER(Tx3)*GBETA(3x1) + Error   #
        #################################################################

        Y = DEP - F + np.matmul(DER,IBETA)

        GBETA = np.linalg.inv(DER.T@DER)@DER.T@Y;
        diffnlls = np.matmul(np.ones((1,3)),np.abs(GBETA-IBETA))

        if diffnlls < prec:
            HALT = 1

        it = it + 1

    return GBETA