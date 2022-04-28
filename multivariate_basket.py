# -*- coding: utf-8 -*-
"""
Created on Mon Jan 17 14:06:06 2022

@author: XYZW
"""

def multivariate_terminal_value(S,K,r,sig,T,ns,ro,qs):
    """
    Simulate the multivariate value of (S_T^1,S_T^2,...,S_T^k)
    """
    import numpy as np
    import scipy.stats as stats
    arr = np.array([0]*len(S))
    multivar_norm = stats.multivariate_normal.rvs(arr,ro,ns)
    smpl = [np.array(S)*np.exp((r-qs-sig**2/2)*T)*np.exp(sig*np.sqrt(T)*x) \
            for x in multivar_norm]
    smpl = np.array(smpl)
    return smpl

def multivariate_basket(S,K,r,sig,T,ns,ro,qs,option = 'call'):
    "Price and delta of a multivariate basket option"
    import numpy as np
    import scipy.stats as stats
    arr = np.array([0]*len(S))
    multivar_norm = stats.multivariate_normal.rvs(arr,ro,ns)
    smpl = [np.array(S)*np.exp((r-np.array(qs)-sig**2/2)*T) * np.exp(sig*np.sqrt(T)*x) \
            for x in multivar_norm]
    smpl = np.array(smpl)
    if option == 'call':
        payoffs = [(np.mean(smpl[i,:])-K)*(np.mean(smpl[i,:])>K) for i in range(ns)]
        delta_aux = [np.mean([smpl[i,j]/(len(S)*S[j])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T) \
             for j in range(len(S))]
    elif option == 'put':
        payoffs = [(-np.mean(smpl[i,:])+K)*(np.mean(smpl[i,:])<K) for i in range(ns)]
        delta_aux = [-np.mean([smpl[i,j]/(len(S)*S[j])*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T) \
             for j in range(len(S))]
    delta = [sum(delta_aux*ro[i,:]*sig*S)/(sig[i]*S[i]) for i in range(len(S))]
    price = np.mean(payoffs)*np.exp(-r*T)
    return price,delta

def multivariate_basket2(S,K,r,sig,T,coeffs,ns,ro,qs,option = 'call'):
    import numpy as np
    import scipy.stats as stats
    arr = np.array([0]*len(S))
    multivar_norm = stats.multivariate_normal.rvs(arr,ro,ns)
    smpl = [np.array(S)*np.exp((r-np.array(qs)-sig**2/2)*T) * np.exp(sig*np.sqrt(T)*x) \
            for x in multivar_norm]
    smpl = np.array(smpl)
    if option == 'call':
        payoffs = [(np.dot(smpl[i,:],coeffs)-K)*(np.dot(smpl[i,:],coeffs)>K) for i in range(ns)]
        delta_aux = [np.mean([smpl[i,j]/S[j] * coeffs[j]*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T) \
             for j in range(len(S))]
    elif option == 'put':
        payoffs = [(-np.dot(smpl[i,:],coeffs)+K)*(np.dot(smpl[i,:],coeffs)<K) for i in range(ns)]
        delta_aux = [-np.mean([smpl[i,j]/S[j] * coeffs[j]*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T) \
             for j in range(len(S))]
    delta = [sum(delta_aux*ro[i,:]*sig*S)/(sig[i]*S[i]) for i in range(len(S))]
    price = np.mean(payoffs)*np.exp(-r*T)
    return price,delta

def multivariate_basket3(S,sig,smpl,K,r,T,coeffs,ro,option = 'call'):
    import numpy as np
    ns = np.size(smpl,0)
    if option in ['call','Call']:
        payoffs = [(np.dot(smpl[i,:],coeffs)-K)*(np.dot(smpl[i,:],coeffs)>K) for i in range(ns)]
        delta_aux = [np.mean([smpl[i,j]/S[j] * coeffs[j]*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T) \
             for j in range(len(S))]
    elif option in [ 'put','Put']:
        payoffs = [(-np.dot(smpl[i,:],coeffs)+K)*(np.dot(smpl[i,:],coeffs)<K) for i in range(ns)]
        delta_aux = [-np.mean([smpl[i,j]/S[j] * coeffs[j]*(payoffs[i]>0) for i in range(ns)])*np.exp(-r*T) \
             for j in range(len(S))]
    delta = [sum(delta_aux*ro[i,:]*sig*S)/(sig[i]*S[i]) for i in range(len(S))]
    price = np.mean(payoffs)*np.exp(-r*T)
    return price,delta

def option_on_maximum(S,sig,smpl,K,r,T,ro,ns,option = 'call'):
    import numpy as np
    if option  == 'call':
        payoffs = [(max(smpl[i,:])-K)*(max(smpl[i,:])>K) for i in range(ns)]
    else:
        payoffs = [(K - max(smpl[i,:]))*(max(smpl[i,:])<K) for i in range(ns)]
    delta = [np.mean([smpl[i,j]/S[j] * (smpl[i,j] == max(smpl[i,:])) * (payoffs[i]>0) \
                      for i in range(ns)]) * np.exp(-r*T) for j in range(len(S))]
    return np.mean(payoffs)*np.exp(-r*T),delta

def option_on_minimum(S,sig,smpl,K,r,T,ro,ns,option = 'put'):
    import numpy as np
    if option  == 'call':
        payoffs = [(min(smpl[i,:])-K)*(min(smpl[i,:])>K) for i in range(ns)]
    else:
        payoffs = [(K - min(smpl[i,:]))*(min(smpl[i,:])<K) for i in range(ns)]
    delta = [np.mean([smpl[i,j]/S[j] * (smpl[i,j] == min(smpl[i,:])) * (payoffs[i]>0) \
                      for i in range(ns)]) * np.exp(-r*T) for j in range(len(S))]
    return np.mean(payoffs)*np.exp(-r*T),delta

def sim_traj_term_val(S,sig,ro,ns,r,qs,T):
    import numpy as np
    import scipy.stats as stats
    arr = np.array([0]*len(S))
    multivar_norm = stats.multivariate_normal.rvs(arr,ro,ns)
    smpl = [np.array(S)*np.exp((r-np.array(qs)-sig**2/2)*T) * np.exp(sig*np.sqrt(T)*x) \
            for x in multivar_norm]
    smpl = np.array(smpl)
    return smpl


def best_of_option(S,sig,smpl,Ks,r,T,ro,ns,option = 'call'):
    "Compute the price and delta of a option whose payoff is"
    "max((S_1-K_1)*1(S1>K1),...,(Sn-Kn)*1(Sn>Kn))"
    import numpy as np
    if option == 'call':
        call_payoffs = [[(smpl[i,j] - Ks[j])*(smpl[i,j]>Ks[j]) for j in range(len(S))] \
                         for i in range(ns)]
        option_payoffs = [max(call_payoffs[i]) for i in range(ns)]
        delta = [np.mean([smpl[i,j]/S[j] * (call_payoffs[i][j] == option_payoffs[i])\
                          *(option_payoffs[i]!=0) for i in range(ns)]) * np.exp(-r*T) for j in range(len(S))]
    else:
        put_payoffs = [[(Ks[j] - smpl[i,j] )*(smpl[i,j]<Ks[j]) for j in range(len(S))] \
                         for i in range(ns)]
        option_payoffs = [max(put_payoffs[i]) for i in range(ns)]
        delta = [np.mean([smpl[i,j]/S[j] * (put_payoffs[i][j] == option_payoffs[i])\
                          *(option_payoffs[i]!=0) for i in range(ns)]) * np.exp(-r*T) for j in range(len(S))]
    price = np.mean(option_payoffs)*np.exp(-r*T)
    return price,delta
        
#%%

