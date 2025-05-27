def no_scaling(lss,**kwargs): #test
    try:
        lss = lss#[:,0]
    except:
        pass
    #eps = 1e-6
    #lss = eps + (1 - 2 * eps) * lss
    return lss