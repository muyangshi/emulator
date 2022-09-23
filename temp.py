#%%
# imports
from model_sim import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import pairwise_distances
from multiprocessing import Pool
import itertools
import time
import matplotlib.pyplot as plt

#%%
# Utilities
F_X_star = lib.pmixture_C
quantile_F_X_star = lib.qRW_newton_C

F_X = lib.F_X_cheat
quantile_F_X = lib.quantile_F_X

f_X = lib.f_X

gaussian_pdf = scipy.stats.norm.pdf
gaussian_cdf = scipy.stats.norm.cdf

GEV_pdf = scipy.stats.genextreme.pdf
GEV_cdf = scipy.stats.genextreme.cdf

def F_Y(y,mu,sigma,ksi):
    # return(GEV_cdf(y,c=-ksi,loc=mu,scale=sigma))
    return(genpareto.cdf(y,c=ksi,loc=mu,scale=sigma))

def f_Y(y,mu,sigma,ksi):
    # return(GEV_pdf(y,c=-ksi,loc=mu,scale=sigma))
    return(genpareto.pdf(y,c=ksi,loc=mu,scale=sigma))

def censored_likelihood(y, u, X_star, p, phi, gamma, tau, mu, sigma, ksi):
    if y <= u:
        return gaussian_cdf((quantile_F_X(p, phi, gamma, tau)-X_star)/tau)
    else: # y > u
        return gaussian_pdf(quantile_F_X(F_Y(y, mu, sigma, ksi), phi, gamma, tau), scale=tau)* \
            f_Y(y, mu, sigma, ksi)/f_X(p, phi, gamma, tau, quantile_F_X(F_Y(y, mu, sigma, ksi),phi, gamma, tau))

def log_censored_likelihood(y, u, X_star, p, phi, gamma, tau, mu, sigma, ksi):
    if y <= u:
        return scipy.stats.norm.logcdf((quantile_F_X(p, phi, gamma, tau)-X_star)/tau)
    else: # y > u
        return scipy.stats.norm.logpdf(quantile_F_X(F_Y(y, mu, sigma, ksi), phi, gamma, tau), scale=tau)+ \
            np.log(f_Y(y, mu, sigma, ksi)) - np.log(f_X(p, phi, gamma, tau, quantile_F_X(F_Y(y, mu, sigma, ksi),phi, gamma, tau)))

#%%
# Some More Utilities
def norm_to_Pareto(z):
    if(isinstance(z, (int, np.int64, float))): z=np.array([z])
    tmp = scipy.stats.norm.cdf(z)
    if np.any(tmp==1): tmp[tmp==1]=1-1e-9
    return 1/(1-tmp)-1

# m is delta, s is gamma in Stable(alpha, 1, gamma, delta)
def rlevy(n, m = 0, s = 1):
  if np.any(s < 0):
    sys.exit("s must be positive")
  return s/scipy.stats.norm.ppf(1-uniform.rvs(0,1,n)/2)**2 + m

def dlevy(r, m=0, s=1, log=False):
    if np.any(s < 0):
        sys.exit("s must be positive")
    if np.any(r < m):
        sys.exit("y must be greater than m")
        
    tmp = np.log(s/(2 * np.pi))/2 - 3 * np.log(r - m)/2 - s/(2 * (r - m))
    if not log: 
        tmp = np.exp(tmp)
    return tmp

## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##          Simulate Dataset Check Smoothness of Censored Likelihood
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##

# %%
# Simulation Parameters
np.random.seed(774759861)
n_cores = 50

# Spatial-Temporal Parameters
N = 60 # number of time replicates
id_x = np.linspace(1,25,num=25) # x locations
id_y = np.linspace(1,8,num=8) # y locations
num_sites = len(id_x)*len(id_y)
id_t = np.arange(N) # time replicates

# Gaussian Process
tau = np.sqrt(10) # standard deviation of the Gaussian nugget terms
tau_square = 10
covariance_matrix = 1 # the Cov for multivariate gaussian Z(s)
rho = 0.01 # the rho in matern kernel exp(-rho * x)
length_scale = 1/rho # scikit/learn parameterization (length_scale)
nu = 0.5 # exponential kernel for matern with nu = 1/2

# Scale Mixture Parameters
phi = 0.5 # the phi in R^phi*W
gamma = 1 # is this the gamma that goes in rlevy?
delta = 0

ksi = 0 # Generalized Parato Distribution parameter for Y
mu = 0 # Generalized Parato Distribution parameter for Y
sigma = 1 # Generalized Parato Distribution parameter for Y

# Censoring parameters
p = 0.9 # censoring threshold probability
q = quantile_F_X(p, phi, gamma, tau) # high threshold for the X
u = genpareto.ppf(p,c=ksi, loc=q, scale=sigma) # censoring threshold value for the Y


# %% 
# Generate Z(s) Gaussian Process, 25 locations, N time replicates
# ---------------------------------------------------------------
# len(id_x) * len(id_y) locations in a 2D array
# sites is indexed (say 25 sites for example) in the following order
"""
    5, 10, 15, 20, 25
    4,  9, 14, 19, 24
    3,  8, 13, 18, 23
    2,  7, 12, 17, 22
    1,  6, 11, 16, 21
"""
s_xy = np.vstack(np.meshgrid(id_x,id_y,indexing='ij')).reshape(2,-1).T

# num_sites by num_sites euclidean distance matrix, entry (i,j) is h(point_i,point_j)
distance_matrix = pairwise_distances(s_xy, metric="euclidean")

# covariance kernel, with nu=0.5, it's exp(-rho*x)
K = Matern(nu=nu,length_scale=length_scale)(0, distance_matrix.ravel()[:,np.newaxis]).reshape(distance_matrix.shape)

# Generate N time replicates, Z.shape = (N,num_sites)
# N time replicates, each row is a time replicate of num_sites sites
# each column is one site, across N time replicates
Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),cov=K,size=N)

# site 17 and site 25 (in blue and green) are closer, so they're more correlated
# they are far from site 200 (red line), so less correlated
# plt.plot(id_t,Z[:,16], 'b.-') 
# plt.plot(id_t,Z[:,24], 'g.-')
# plt.plot(id_t,Z[:,199],'r.-')

# plot the difference between the sites
# plt.plot(id_t,Z[:,16] - Z[:,24],'b.-')
# plt.plot(id_t,Z[:,16] - Z[:,199],'r.-')

# d = pairwise_distances([[1,0],[3,0],[5,0]], metric="euclidean")
# K = Matern(nu=0.5, length_scale=1/0.2)(0, d.ravel()[:, np.newaxis]).reshape(d.shape)


# For 3D plotting
t = 1
xyz_t = np.insert(arr=s_xy,obj=2,values=Z[t,:],axis=1)
fig = plt.figure()
ax = plt.axes(projection='3d')
points = ax.scatter(xyz_t[:,0], xyz_t[:,1], xyz_t[:,2],color='slateblue',marker=".")
surf = ax.plot_trisurf(xyz_t[:,0], xyz_t[:,1], xyz_t[:,2],color='grey',alpha=0.8)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Z")
plt.title("time = " + str(t))
plt.show()