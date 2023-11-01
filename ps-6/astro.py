#!/usr/bin/env python
# coding: utf-8

# In[1]:


import astropy.io.fits
import matplotlib.pyplot as plt
import numpy as np
hdu_list = astropy.io.fits.open('specgrid.fits')
logwave = hdu_list['LOGWAVE'].data
flux = hdu_list['FLUX'].data


# In[2]:


#problem a
wave=(logwave)
plt.plot(wave,flux[0],label='galaxy 1')
plt.plot(wave,flux[10],label='galaxy 11')
plt.plot(wave,flux[20],label='galaxy 21')
plt.xlabel("log($\lambda(\AA)$)")
plt.ylabel('Flux($10^{-17}erg\ s^{-1} cm^{-2}\AA^{-1}$)')
plt.legend()
plt.title("Flux of 3 different galaxies")
plt.savefig("problema.png",dpi=1000)


# In[3]:


#problem b
flux_sum = np.sum(flux, axis = 1)
flux_normalized = flux/np.tile(flux_sum, (np.shape(flux)[1], 1)).T
plt.plot(wave,flux_normalized[0],label='galaxy 1')
plt.plot(wave,flux_normalized[10],label='galaxy 11')


# In[4]:


#problem c
means_normalized = np.mean(flux_normalized, axis=1)
flux_normalized_0_mean = flux_normalized-np.tile(means_normalized, (np.shape(flux)[1], 1)).T
plt.plot(wave,flux_normalized_0_mean[0],label='galaxy 1')
plt.xlabel("log($\lambda(\AA)$)")
plt.savefig("problemc.png",dpi=1000)


# In[5]:


#problem d
C = np.dot(np.transpose(flux_normalized_0_mean),flux_normalized_0_mean)
C.shape
eigvals,eigvecs=np.linalg.eig(C)
sort_idx = np.argsort(eigvals)[::-1]
cn1=np.max(eigvals)/np.min(eigvals)
print(cn1)
eigvecs = eigvecs[:, sort_idx]
eigvals = eigvals[sort_idx]
for i in range(5):
    plt.plot(eigvecs[i])
plt.savefig("problemd.png",dpi=1000)


# In[6]:


U, S, Vh = np.linalg.svd(flux_normalized_0_mean, full_matrices=False)
# rows of Vh are eigenvectors
eigvecs_svd = Vh.T
eigvals_svd = S**2
cn2=np.max(S**2)/np.min(S**2)
print(cn2)
svd_sort = np.argsort(eigvals_svd)[::-1]
eigvecs_svd = eigvecs_svd[:,svd_sort]
eigvals_svd = eigvals_svd[svd_sort]
for i in range(5):
    plt.plot(eigvecs_svd[i])
plt.savefig("probleme.png",dpi=1000)


# In[7]:


def sorted_eigs(r, return_eigvalues = False):
    """
    Calculate the eigenvectors and eigenvalues of the correlation matrix of r
    -----------------------------------------------------
    """
    corr=r.T@r
    eigs=np.linalg.eig(corr) #calculate eigenvectors and values of original 
    arg=np.argsort(eigs[0])[::-1] #get indices for sorted eigenvalues
    eigvec=eigs[1][:,arg] #sort eigenvectors
    eig = eigs[0][arg] # sort eigenvalues
    if return_eigvalues == True:
        return eig, eigvec
    else:
        return eigvec


# In[13]:


def PCA(l, r, project = True):
    """
    Perform PCA dimensionality reduction
    --------------------------------------------------------------------------------------
    """
    eigvector = sorted_eigs(r)
    eigvec=eigvector[:,:l] #sort eigenvectors, only keep l
    reduced_wavelength_data= np.dot(eigvec.T,r.T) #np.dot(eigvec.T, np.dot(eigvec,r.T))
    #print(reduced_wavelength_data[0,:])
    plt.plot(reduced_wavelength_data[0,:], reduced_wavelength_data[1,:],'.',label = 'c1')
    plt.plot(reduced_wavelength_data[0,:], reduced_wavelength_data[2,:],'.',label = 'c2')
    plt.legend()
    plt.savefig("problemh.png",dpi=1000)
    plt.clf()
    if project == False:
        return reduced_wavelength_data.T # get the reduced wavelength weights
    else: 
        return np.dot(eigvec, reduced_wavelength_data).T # multiply eigenvectors by 
                                                        # weights to get approximate spectrum


# In[14]:


# check to make sure that using all eigenvectors reconstructs original signal
plt.plot(logwave, PCA(5,flux_normalized_0_mean)[1,:], label = 'l = 5')
plt.plot(logwave, flux_normalized_0_mean[1,:], label = 'original data')
plt.xlabel("log($\lambda(\AA)$)")
plt.legend()
plt.savefig("problemg.png",dpi=1000)


# In[10]:


s=np.zeros(20)
for i in range(1,21):
    #print(np.sum((PCA(i,flux_normalized_0_mean)[1,:]-flux_normalized_0_mean[1,:])**2))
    s[i-1]=np.sum((PCA(i,flux_normalized_0_mean)[1,:]-flux_normalized_0_mean[1,:])**2)
plt.plot(range(1,21), s)
plt.xlabel("N")
plt.savefig("problemi.png",dpi=1000)


# In[ ]:




