# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 14:16:09 2022

@author: XYZW
"""
from multivariate_gbm import multivariate_gbm
def multivariate_basket_average(S,K,r,sigs,T,ro,ns,M,qs,option = 'call'):
    import numpy as np
    gbm = multivariate_gbm(S,K,T,r,sigs,ns,M,ro,qs)
    if option == 'call':
        payoffs = [(np.mean([np.mean(gbm[k][i,:]) for k in range(len(S))])-K)*\
               (np.mean([np.mean(gbm[k][i,:]) for k in range(len(S))])>K) for i in range(ns)]
        delta_aux = [np.mean([np.mean(gbm[k][i,:])/(len(S)*S[k])*(payoffs[i]>0) \
                      for i in range(ns)])*np.exp(-r*T) for k in range(len(S))]
    elif option == 'put':
        payoffs = [(np.mean([-np.mean(gbm[k][i,:]) for k in range(len(S))])+K)*\
               (np.mean([np.mean(gbm[k][i,:]) for k in range(len(S))])<K) for i in range(ns)]
        delta_aux = [np.mean([-np.mean(gbm[k][i,:])/(len(S)*S[k])*(payoffs[i]>0) \
                      for i in range(ns)])*np.exp(-r*T) for k in range(len(S))]
    delta = [sum(delta_aux*ro[i,:]*sigs*S)/(sigs[i]*S[i]) for i in range(len(S))]
    price = np.exp(-r*T)*np.mean(payoffs)
    return price,delta

def multivariate_basket_average2(S,K,r,sigs,T,ro,gbm,option = 'call'):
    import numpy as np
    if option == 'call':
        payoffs = [(np.mean([np.mean(gbm[k][i,:]) for k in range(len(S))])-K)*\
               (np.mean([np.mean(gbm[k][i,:]) for k in range(len(S))])>K) for i in range(ns)]
        delta_aux = [np.mean([np.mean(gbm[k][i,:])/(len(S)*S[k])*(payoffs[i]>0) \
                      for i in range(ns)])*np.exp(-r*T) for k in range(len(S))]
    elif option == 'put':
        payoffs = [(np.mean([-np.mean(gbm[k][i,:]) for k in range(len(S))])+K)*\
               (np.mean([np.mean(gbm[k][i,:]) for k in range(len(S))])<K) for i in range(ns)]
        delta_aux = [np.mean([-np.mean(gbm[k][i,:])/(len(S)*S[k])*(payoffs[i]>0) \
                      for i in range(ns)])*np.exp(-r*T) for k in range(len(S))]
    delta = [sum(delta_aux*ro[i,:]*sigs*S)/(sigs[i]*S[i]) for i in range(len(S))]
    price = np.exp(-r*T)*np.mean(payoffs)
    return price,delta