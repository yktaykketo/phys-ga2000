#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimize


# In[2]:


def p(x, beta_0, beta_1):
    return 1/(1+np.exp(-(beta_0+beta_1*x)))


# In[3]:


import pandas as pd
data = pd.read_csv('survey.csv')  
xs = data['age'].to_numpy()
ys = data['recognized_it'].to_numpy()
x_sort = np.argsort(xs)
age = xs[x_sort]
recognize = ys[x_sort]


# In[4]:


def log_likelihood(beta, xs, ys):
    beta_0 = beta[0]
    beta_1 = beta[1]
    epsilon = 1e-16
    l_list = [ys[i]*np.log(p(xs[i], beta_0, beta_1)/(1-p(xs[i], beta_0, beta_1)+epsilon)) 
              + np.log(1-p(xs[i], beta_0, beta_1)+epsilon) for i in range(len(xs))]
    ll = np.sum(np.array(l_list), axis = -1)
    return -ll # return log likelihood


# In[5]:


def Covariance(hess_inv, resVariance):
    return hess_inv * resVariance
#Error of parameters
def error(hess_inv, resVariance):
    covariance = Covariance(hess_inv, resVariance)
    return np.sqrt( np.diag( covariance ))
pstart = [0.5,0.5]
beta = np.array([-1,1])


# In[6]:


result = optimize.minimize(lambda beta, age, recognize: log_likelihood(beta, age, recognize),beta,args=(age, recognize))
hess_inv = result.hess_inv # inverse of hessian matrix
var = result.fun/(len(recognize)-len(pstart))
dFit = error(hess_inv,var)
print('Optimal parameters and error:\n\tp: ' , result.x, '\n\tdp: ', dFit)
print('Covariance matrix of optimal parameters:\n\tC: ' , Covariance( hess_inv,  var))


# In[7]:


plt.plot(age, p(age, result.x[0], result.x[1]), label='Logistic Function')
plt.plot(age, recognize, 'o', label='Data')
plt.title('Probability of hearing the phrase')
plt.xlabel('Age (y)')
plt.ylabel('Probabilty')
plt.legend()
plt.savefig('problem2.png')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




