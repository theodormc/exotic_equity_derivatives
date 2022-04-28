# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 22:35:26 2022

@author: XYZW
"""
def gbm_traj_bivariate(S1,S2,r,sig1,sig2,T,ns,M,ro,q1=0,q2=0):
    import numpy as np
    import scipy.stats as stats
    times = np.linspace(0,T,M+1)
    arr,cov = [0,0],[[1,ro],[ro,1]]
    gbm1,gbm2 = [],[]
    for i in range(ns):
        smpl = stats.multivariate_normal.rvs(arr,cov,M)
        traj1,traj2 = [S1],[S2]
        for j in range(len(times)-1):
            traj1.append(traj1[-1]+(r-q1)*traj1[-1]*(times[j+1]-times[j])+\
                        sig1*traj1[-1]*np.sqrt(times[j+1]-times[j])*smpl[j,0])
            traj2.append(traj2[-1]+(r-q2)*traj2[-1]*(times[j+1]-times[j])+\
                        sig2*traj2[-1]*np.sqrt(times[j+1]-times[j])*smpl[j,1])
        gbm1.append(traj1)
        gbm2.append(traj2)
    return gbm1,gbm2 

def basket_barrier_option(S1,S2,H1,H2,K,T,r,sig1,sig2,ns,M,ro):
    gbm1,gbm2 = gbm_traj_bivariate(S1,S2,r,sig1,sig2,T,ns,M,ro)
    import numpy as np
    gbm1 = np.array(gbm1)
    gbm2 = np.array(gbm2)
    payoffs = [(min(gbm1[i,:])<H1)*(min(gbm2[i,:])<H2)*((gbm1[i,-1]+gbm2[i,-1])/2-K)*\
               ((gbm1[i,-1]+gbm2[i,-1])/2>K) for i in range(ns)]
    price = np.mean(payoffs)*np.exp(-r*T)
    delta1 = np.mean(np.array(payoffs)/S1)*np.exp(-r*T)
    delta2 = np.mean(np.array(payoffs)/S2)*np.exp(-r*T)
    return price,delta1,delta2



#%%
        