def lfoc(X,zt,ki,kj,alf,a,delta):
    # This function solves optial labor policy from intratemporal condition 
    # and RC in the dynamic programming system.
    
    c = zt*ki**alf*X**(1-alf)+(1-delta)*ki-kj
    F = c/(1-X)-(1-alf)/a*zt*ki**alf*X**(-alf)

    return F
