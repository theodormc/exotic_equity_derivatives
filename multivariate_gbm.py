# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:15:29 2022

@author: XYZW
"""

def multivariate_gbm(S,K,T,r,sigs,ns,M,ro,qs):
    import scipy.stats as stats
    import numpy as np
    dt = T/M
    arr = np.array([0]*len(S))
    gbm = np.zeros((len(S),ns,M+1))
    for i in range(ns):
        smpl = stats.multivariate_normal.rvs(arr,ro,M) 
        for k in range(len(S)):
            gbm[k][i,:] = np.cumprod([S[k]]+[1+(r-qs[k])*dt+\
                   sigs[k]*np.sqrt(dt)*smpl[j,k] for j in range(M)])
    
    return gbm

#%%
def multivariate_gbm2(S,T,mu,sigs,ns,M,ro,qs):
    import scipy.stats as stats
    import numpy as np
    dt = T/M
    arr = np.array([0]*len(S))
    gbm = np.zeros((len(S),ns,M+1))
    for i in range(ns):
        smpl = stats.multivariate_normal.rvs(arr,ro,M) 
        for k in range(len(S)):
            gbm[k][i,:] = np.cumprod([S[k]]+[1+(mu[k]-qs[k])*dt+\
                   sigs[k]*np.sqrt(dt)*smpl[j,k] for j in range(M)])
    return gbm

#%%

