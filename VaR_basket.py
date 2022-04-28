# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 17:51:32 2022

@author: XYZW
"""
from basket_options import basket_option_averages
def Value_at_Risk_basket(S1,S2,K1,K2,K,n1,n2,sig1,sig2,r,T,ro,h = 10/252,alpha = 0.95,no_opt1 = [10,10],\
                  no_opt2 = 10):
    from option_prices_MC import price_BS
    import numpy as np
    import scipy.stats as stats
    def std_dev_port(exposures,cov_mat):
        return np.sqrt(np.dot(np.dot(exposures,cov_mat),exposures.T))
    exposures1 = np.array([S1,S2])*np.array([n1,n2])
    cov_mat = np.array([sig1,sig2]).T*np.array([sig1,sig2])*np.array([[1,ro],[ro,1]])
    price1,delta1 = price_BS(S1,K1,r,sig1,T,option = 'put')
    price2,delta2 = price_BS(S2,K2,r,sig2,T,option = 'put')
    price_avg,delta_avg1,delta_avg2 = basket_option_averages(S1,S2,K,T,r,sig1,sig2,\
                            10000,20,ro,q1 = 0,q2 = 0,option = 'put')
    exposures2 = lambda x:np.array([S1,S2])*(np.array([n1,n2])+np.array(x)*np.array([delta1,delta2]))
    exposures3 = lambda x:np.array([S1,S2])*(np.array([n1,n2])+x*np.array([delta_avg1,delta_avg2]))
    VaR1 = stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev_port(exposures1,cov_mat)
    VaR2 = stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev_port(exposures2(no_opt1),cov_mat)
    VaR3 = stats.norm.isf(1-alpha)*np.sqrt(h)*std_dev_port(exposures3(no_opt2),cov_mat)
    return VaR1,VaR2,VaR3,price_avg