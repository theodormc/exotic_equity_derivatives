# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 08:25:00 2022

@author: XYZW
"""
from parisian_options import gbm_traj
def test_parisian():
    S0,H,K,r,sig,T,ns,M,q = 100,112,100,0.02,0.3,1,1000,52,0
    print(parisian_option(S0,H,K,r,sig,T,ns,M,q))
    from option_prices_MC import price_BS
    print(price_BS(S0,K,r,sig,T,q))
test_parisian()
#%%
def test_gbm():
    import numpy as np
    S0,r,sig,T,ns,M,q = 100,0.02,0.3,1,10,10,0
    gbm = gbm_traj(S0,r,sig,T,ns,M)
    print(np.mean(gbm[:,-1]),np.exp(r*T)*S0)
    l = [[1,2,3],[1,2]]
    print(l)
    import scipy.stats as stats
    smpl = stats.norm.rvs(0,1,20)
    log_smpl = S0*np.exp((r-sig**2/2)*T+sig*np.sqrt(T)*smpl)
    print(log_smpl)
    indices = [i for i,x in enumerate(log_smpl) if x>100]
    print(indices)
    def consec_five(l):
        for i in range(len(l)-5):
            if [l[i+j+1]-l[i+j] for j in range(0,5)]==[1,1,1,1,1]:
                return 1
        return 0
    print(consec_five(indices))
    print(gbm)
    print(gbm[1,5:10])
    print(np.mean(gbm[1,6:10]))
    print(np.linspace(0,1,10))
test_gbm()