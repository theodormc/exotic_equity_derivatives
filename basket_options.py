# -*- coding: utf-8 -*-
"""
Created on Wed Jan  5 10:09:39 2022

@author: XYZW
"""
def bivariate_gbm(S1,S2,r,sig1,sig2,T,ns,M,ro,q1=0,q2 = 0):
    import scipy.stats as stats
    import numpy as np
    dt = T/M
    gbm1,gbm2 = [],[]
    arr = np.array([0,0])
    cov = np.array([[1,ro],[ro,1]])
    for i in range(ns):
        smpl = stats.multivariate_normal.rvs(arr,cov,M)
        traj1,traj2 = [S1],[S2]
        for j in range(M-1):
            traj1.append(traj1[-1]+(r-q1)*traj1[-1]*dt+sig1*traj1[-1]*np.sqrt(dt)*smpl[j,0])
            traj2.append(traj2[-1]+(r-q2)*traj2[-1]*dt + sig2*traj2[-1]*np.sqrt(dt)*smpl[j,1])
        gbm1.append(traj1)
        gbm2.append(traj2)
    gbm1,gbm2 = np.array(gbm1),np.array(gbm2)
    return gbm1,gbm2
#%%
def basket_option_averages(S1,S2,K,T,r,sig1,sig2,ns,M,ro,q1 = 0,q2 = 0,option = 'call'):
    import numpy as np
    gbm1,gbm2 = bivariate_gbm(S1,S2,r,sig1,sig2,T,ns,M,ro,q1,q2)
    if option=='call':
        payoffs = [max((gbm1[i,-1]+gbm2[i,-1])/2-(np.mean(gbm1[i,:])+np.mean(gbm2[i,:]))/2,0) for i in range(ns)]
        delta1 = np.mean([gbm1[i,-1]/(2*S1) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
        delta2 = np.mean([gbm2[i,-1]/(2*S2) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    elif option == 'put':
        payoffs = [max(-(gbm1[i,-1]+gbm2[i,-1])/2+(np.mean(gbm1[i,:])+np.mean(gbm2[i,:]))/2,0) for i in range(ns)]
        delta1 = np.mean([-gbm1[i,-1]/(2*S1) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
        delta2 = np.mean([-gbm2[i,-1]/(2*S2) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    price = np.mean(payoffs)*np.exp(-r*T)
    return price,delta1,delta2
#%%
from trivariate_basket_average import trivariate_gbm
def basket_option_averages2(S,K,T,r,sigs,ns,M,ro,qs = [0,0,0]):
    "IT IS A FLOATING STRIKE TRIVARIATE AVERAGE OPTION"
    import numpy as np
    gbm1,gbm2,gbm3 = trivariate_gbm(S,sigs,r,T,ns,M,ro,qs)
    payoffs = [max((gbm1[i,-1]+gbm2[i,-1]+gbm3[i,-1])/3-\
                   (np.mean(gbm1[i,:])+np.mean(gbm2[i,:])+np.mean(gbm3[i,:]))/3,0) for i in range(ns)]
    price = np.mean(payoffs)*np.exp(-r*T)
    delta1 = np.mean([gbm1[i,-1]/(3*S[0]) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    delta2 = np.mean([gbm2[i,-1]/(3*S[1]) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    delta3 = np.mean([gbm3[i,-1]/(3*S[2]) * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    return price,delta1,delta2,delta3

def basket_option_averages3(S,K,coeffs,T,r,sigs,ns,M,ro,qs = [0,0,0],option = 'call'):
    "FLOATING STRIKE /AVERAGE STRIKE OPTION PRICING"
    import numpy as np
    gbm1,gbm2,gbm3 = trivariate_gbm(S,sigs,r,T,ns,M,ro,qs)
    if option=='call':
        payoffs = [max((gbm1[i,-1]*coeffs[0]+gbm2[i,-1]*coeffs[1]+gbm3[i,-1]*coeffs[2])-\
                   (np.mean(gbm1[i,:])*coeffs[0]+np.mean(gbm2[i,:])*coeffs[1]+\
                    np.mean(gbm3[i,:])*coeffs[2]),0) for i in range(ns)]
    elif option=='put':
        payoffs = [max(-(gbm1[i,-1]*coeffs[0]+gbm2[i,-1]*coeffs[1]+gbm3[i,-1]*coeffs[2])+\
                   (np.mean(gbm1[i,:])*coeffs[0]+np.mean(gbm2[i,:])*coeffs[1]+\
                    np.mean(gbm3[i,:])*coeffs[2]),0) for i in range(ns)]
    price = np.mean(payoffs)*np.exp(-r*T)
    delta1 = np.mean([gbm1[i,-1]/S[0] * coeffs[0] * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    delta2 = np.mean([gbm2[i,-1]/S[1] * coeffs[1] * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    delta3 = np.mean([gbm3[i,-1]/S[2] * coeffs[2] * (payoffs[i]>0) for i in range(ns)])*np.exp(-r*T)
    return price,delta1,delta2,delta3
#%%
def spread_option(S1,S2,K,T,r,sig1,sig2,ns,ro):
    import scipy.stats as stats
    import numpy as np
    arr = np.array([0,0])
    cov = np.array([[1,ro],[ro,1]])
    smpl = stats.multivariate_normal.rvs(arr,cov,ns)
    aux = np.exp(np.sqrt(T)*np.array([smpl[:,0]*sig1,smpl[:,1]*sig2]).T)
    terminal_value = [S1*np.exp((r-sig1**2/2)*T),S2*np.exp((r-sig2**2/2)*T)]*aux
    payoffs_aux = terminal_value[:,0]-terminal_value[:,1]-K
    payoffs = [max([payoffs_aux[i],0]) for i in range(len(payoffs_aux))]
    price = np.mean(payoffs)*np.exp(-r*T)
    return price

def basket_option(S1,S2,K,T,r,sig1,sig2,ns,ro):
    "Price of a call option whose payoff is max((S1+S2)/2-K,0)"
    import scipy.stats as stats
    import numpy as np
    arr = np.array([0,0])
    cov = np.array([[1,ro],[ro,1]])
    smpl = stats.multivariate_normal.rvs(arr,cov,ns)
    aux = np.exp(np.sqrt(T)*np.array([smpl[:,0]*sig1,smpl[:,1]*sig2]).T)
    terminal_value = [S1*np.exp((r-sig1**2/2)*T),S2*np.exp((r-sig2**2/2)*T)]*aux
    payoffs = [max([(terminal_value[i,0]+terminal_value[i,1])/2-K,0]) for i in range(ns)]
    price = np.mean(payoffs)*np.exp(-r*T)
    delta1_sim = 1/2*np.exp((r-sig1**2/2)*T)*np.array([np.exp(sig1*np.sqrt(T)*smpl[i,0]) \
                        *(payoffs[i]>0)for i in range(ns)])
    delta2_sim = 1/2*np.exp((r-sig2**2/2)*T)*np.array([np.exp(sig2*np.sqrt(T)*smpl[i,1]) \
                        *(payoffs[i]>0) for i in range(ns)])
    delta1 = np.exp(-r*T)*np.mean(delta1_sim)
    delta2 = np.exp(-r*T)*np.mean(delta2_sim)
    return price,delta1,delta2


#%%
def call_on_maximum(S1,S2,K,T,r,sig1,sig2,ns,ro):
    import scipy.stats as stats
    import numpy as np
    arr = np.array([0,0])
    cov = np.array([[1,ro],[ro,1]])
    smpl = stats.multivariate_normal.rvs(arr,cov,ns)
    aux = np.exp(np.sqrt(T)*np.array([smpl[:,0]*sig1,smpl[:,1]*sig2]).T)
    terminal_value = [S1*np.exp((r-sig1**2/2)*T),S2*np.exp((r-sig2**2/2)*T)]*aux
    payoffs = [max([max([terminal_value[i,0],terminal_value[i,1]])-K,0]) for i in range(ns)]
    price = np.mean(payoffs)*np.exp(-r*T)
    delta1_sim = [np.exp((r-sig1**2/2)*T+sig1*np.sqrt(T)*smpl[i,0])*\
                  (payoffs[i]>0)*(terminal_value[i,0]>terminal_value[i,1]) for i in range(ns)]
    delta2_sim = [np.exp((r-sig2**2/2)*T+sig2*np.sqrt(T)*smpl[i,1])*\
                  (payoffs[i]>0)*(terminal_value[i,1]>terminal_value[i,0]) for i in range(ns)]
    delta1 = np.mean(delta1_sim)*np.exp(-r*T)
    delta2 = np.mean(delta2_sim)*np.exp(-r*T)
    return price,delta1,delta2

#print(call_on_maximum(100,110,112,1,0.02,0.3,0.2,1000,0.5))
#%%
def basket_option2(S,K,r,sigs,corr,T,ns,option = 'call'):
    """
    Inputs:
        S: = vector of current stock prices
        K: = the threshold 
        sigs: = vector of idiosyncratic volatilities
        corr: = correlation matrix
    """
    import scipy.stats as stats
    import numpy as np
    arr = np.array([0,0,0])
    smpl = stats.multivariate_normal.rvs(arr,corr,ns)
    terminal_value = S*np.exp(np.sqrt(T)*sigs*smpl+(r-sigs**2/2)*T)
    if option in ["call","Call"]:
        payoffs = [max([np.mean(terminal_value[i,:])-K,0]) for i in range(ns)]
    elif option in ["Put","put"]:
        payoffs = [max([K - np.mean(terminal_value[i,:]),0]) for i in range(ns)]
    price = np.mean(payoffs)*np.exp(-r*T)
    delta_sims = np.array([np.exp((r-sigs**2/2)*T + sigs*np.sqrt(T)*smpl[i,:]) \
                        *(payoffs[i]>0)  for i in range(ns)])
    """delta1_sim = 1/2*np.exp((r-sig1**2/2)*T)*np.array([np.exp(sig1*np.sqrt(T)*smpl[i,0]) \
                        *(payoffs[i]>0)for i in range(ns)])
    delta2_sim = 1/2*np.exp((r-sig2**2/2)*T)*np.array([np.exp(sig2*np.sqrt(T)*smpl[i,1]) \
                        *(payoffs[i]>0) for i in range(ns)])
    delta1 = np.exp(-r*T)*np.mean(delta1_sim)
    delta2 = np.exp(-r*T)*np.mean(delta2_sim)"""
    delta = np.mean(delta_sims,0)
    return price,delta