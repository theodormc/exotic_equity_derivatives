# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:58:16 2022

@author: XYZW
"""

def trivariate_gbm(S,sigs,r,T,ns,M,ro,qs=[0,0,0]):
    import scipy.stats as stats
    import numpy as np
    dt = T/M
    gbm1,gbm2,gbm3 = [],[],[]
    arr = np.array([0,0,0])
    for i in range(ns):
        smpl = stats.multivariate_normal.rvs(arr,ro,M)
        traj1,traj2,traj3 = [S[0]],[S[1]],[S[2]]
        for j in range(M-1):
            traj1.append(traj1[-1]+(r-qs[0])*traj1[-1]*dt+sigs[0]*traj1[-1]*np.sqrt(dt)*smpl[j,0])
            traj2.append(traj2[-1]+(r-qs[1])*traj2[-1]*dt + sigs[1]*traj2[-1]*np.sqrt(dt)*smpl[j,1])
            traj3.append(traj3[-1]+(r-qs[2])*traj3[-1]*dt+sigs[2]*traj2[-1]*np.sqrt(dt)*smpl[j,2])
        gbm1.append(traj1)
        gbm2.append(traj2)
        gbm3.append(traj3)
    gbm1,gbm2,gbm3 = np.array(gbm1),np.array(gbm2),np.array(gbm3)
    return gbm1,gbm2,gbm3

def trivariate_basket_average(S,sigs,K,r,T,ns,M,ro,qs = [0,0,0],option = 'call'):
    gbm1,gbm2,gbm3 = trivariate_gbm(S,sigs,r,T,ns,M,ro,qs)
    import numpy as np
    if option == 'call':
        payoffs = [((np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3-K)*\
               ((np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3>K) for i in range(ns)]
        delta1_aux = np.mean([np.mean(gbm1[i,:])/(3*S[0])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
        delta2_aux = np.mean([np.mean(gbm2[i,:])/(3*S[1])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
        delta3_aux = np.mean([np.mean(gbm3[i,:])/(3*S[2])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    elif option=='put':
        payoffs = [(-(np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3+K)*\
               ((np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3<K) for i in range(ns)]
        delta1_aux = np.mean([-np.mean(gbm1[i,:])/(3*S[0])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
        delta2_aux= np.mean([-np.mean(gbm2[i,:])/(3*S[1])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
        delta3_aux= np.mean([-np.mean(gbm3[i,:])/(3*S[2])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    delta_aux = np.array([delta1_aux,delta2_aux,delta3_aux])
    delta = [sum(delta_aux*ro[i,:]*sigs*S)/(sigs[i]*S[i]) for i in range(len(S))]
    price = np.mean(payoffs)*np.exp(-r*T)
    return price,delta

def trivariate_basket_average_CV(S,sigs,K,r,T,ns,M,ro,qs = [0,0,0],option = 'call'):
    gbm1,gbm2,gbm3 = trivariate_gbm(S,sigs,r,T,ns,M,ro,qs)
    import numpy as np
    from sklearn.linear_model import LinearRegression as linreg
    if option == 'call':
        payoffs1 = np.array([(np.mean(gbm1[i,:])-K)*(np.mean(gbm1[i,:])>K) for i in range(ns)])
        payoffs2 = np.array([(np.mean(gbm2[i,:])-K)*(np.mean(gbm2[i,:])>K) for i in range(ns)])
        payoffs3 = np.array([(np.mean(gbm3[i,:])-K)*(np.mean(gbm3[i,:])>K) for i in range(ns)])
        payoffs_basket_avg = [((np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3-K)*\
               ((np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3>K) for i in range(ns)]
        X = np.array([payoffs1,payoffs2,payoffs3]).T
        y = np.array(payoffs_basket_avg).T
        reg = linreg().fit(X,y)
        payoffs_CV = reg.predict(X)
        price = np.exp(-r*T)*np.mean(payoffs_CV)
    elif option=='put':
        payoffs1 = np.array([(-np.mean(gbm1[i,:])+K)*(np.mean(gbm1[i,:])<K) for i in range(ns)])
        payoffs2 = np.array([(-np.mean(gbm2[i,:])+K)*(np.mean(gbm2[i,:])<K) for i in range(ns)])
        payoffs3 = np.array([(-np.mean(gbm3[i,:])+K)*(np.mean(gbm3[i,:])<K) for i in range(ns)])
        payoffs_basket_avg = [(-(np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3+K)*\
               (-(np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3<K) for i in range(ns)]
        X = np.array([payoffs1,payoffs2,payoffs3]).T
        y = np.array(payoffs_basket_avg).T
        reg = linreg().fit(X,y)
        payoffs_CV = reg.predict(X)
        price = np.exp(-r*T)*np.mean(payoffs_CV)
    return price
        
