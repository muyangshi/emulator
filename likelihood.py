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
rho = 0.2 # the rho in matern kernel exp(-rho * x)
length_scale = 1/rho # scikit/learn parameterization (length_scale)
nu = 0.5 # exponential kernel for matern with nu = 1/2

# Scale Mixture Parameters
phi = 0.5 # the phi in R^phi*W
gamma = 1 # is this the gamma that goes in rlevy?
delta = 0

ksi = 0 # Generalized Parato Distribution parameter for Y
# mu = 0 # Generalized Parato Distribution parameter for Y
sigma = 1 # Generalized Parato Distribution parameter for Y

# Censoring parameters
p = 0.9 # censoring threshold probability
q = 0 # high threshold for the observed data
mu = q

# censoring threshold value for the latent process, X in the code, T(Y) in the paper
# u = quantile_F_X(p, phi, gamma, tau)

#%%
# Utilities
F_X_star = lib.pmixture_C
quantile_F_X_star = lib.qRW_newton_C

F_X = lib.F_X_cheat
quantile_F_X = lib.quantile_F_X

F_X_vec = np.vectorize(F_X,otypes=[float])
def F_X_vec_wrapper(X):
    return(F_X_vec(X, phi, gamma, tau))

f_X = lib.f_X

gaussian_pdf = scipy.stats.norm.pdf
gaussian_cdf = scipy.stats.norm.cdf

GEV_pdf = scipy.stats.genextreme.pdf
GEV_cdf = scipy.stats.genextreme.cdf

def F_Y(y,mu,sigma,ksi):
    # return(GEV_cdf(y,c=-ksi,loc=mu,scale=sigma))
    # return(genpareto.cdf(y,c=ksi,loc=mu,scale=sigma))
    return(p + (1-p)*genpareto.cdf(y, c=ksi, loc=mu, scale=sigma))

def f_Y(y,mu,sigma,ksi):
    # return(GEV_pdf(y,c=-ksi,loc=mu,scale=sigma))
    # return(genpareto.pdf(y,c=ksi,loc=mu,scale=sigma))
    return((1-p)*genpareto.cdf(y, c=ksi, loc=mu, scale=sigma))

def censored_likelihood(y, q, X_star, p, phi, gamma, tau, mu, sigma, ksi):
    if y <= q:
        return gaussian_cdf((quantile_F_X(p, phi, gamma, tau)-X_star)/tau)
    else: # y > u
        return gaussian_pdf(quantile_F_X(F_Y(y, mu, sigma, ksi), phi, gamma, tau), scale=tau)* \
            f_Y(y, mu, sigma, ksi)/f_X(phi, gamma, tau, quantile_F_X(F_Y(y, mu, sigma, ksi),phi, gamma, tau))

def log_censored_likelihood(y, q, X_star, p, phi, gamma, tau, mu, sigma, ksi):
    if y <= q:
        return scipy.stats.norm.logcdf((quantile_F_X(p, phi, gamma, tau)-X_star)/tau)
    else: # y > q
        return scipy.stats.norm.logpdf(quantile_F_X(F_Y(y, mu, sigma, ksi), phi, gamma, tau), scale=tau)+ \
            np.log(f_Y(y, mu, sigma, ksi)) - np.log(f_X(phi, gamma, tau, quantile_F_X(F_Y(y, mu, sigma, ksi),phi, gamma, tau)))

log_censored_likelihood_vec = np.vectorize(log_censored_likelihood, otypes=[float])

def log_censored_likelihood_wrapper(site, phi):
    """
    wrapper function used for parallel process
    reads in site and phi from pool.map()
    the other arguments should be read from the global environment
    """
    # print("phi: ", phi)
    # print("site: ", site)
    site = int(site)
    # print("site int: ", site)
    return(log_censored_likelihood_vec(Y[:,site], q, X_star[:,site], p, phi, gamma, tau, mu, sigma, ksi))

def log_censored_likelihood_wrapper_gamma(site, gamma):
    site = int(site)
    return(log_censored_likelihood_vec(Y[:,site], q, X_star[:,site], p, phi, gamma, tau, mu, sigma, ksi))

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
# t = 1
# xyz_t = np.insert(arr=s_xy,obj=2,values=Z[t,:],axis=1)
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# points = ax.scatter(xyz_t[:,0], xyz_t[:,1], xyz_t[:,2],color='slateblue',marker="o")
# surf = ax.plot_trisurf(xyz_t[:,0], xyz_t[:,1], xyz_t[:,2],color='orange',alpha=0.8)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("Z")
# plt.title("time = " + str(t))
# plt.show()

# %%
# Transform Z(s) to W(s) = g(Z(s))
# --------------------------------
# shpae = (N, num_sites)
W = norm_to_Pareto(Z)

# plt.plot(id_t,W[:,16], 'b.-')
# plt.plot(id_t,W[:,24], 'g.-')
# plt.plot(id_t,W[:,199],'r.-')

# %%
# Generate R(s) Levy
# Generate one single R at each of N time replicate for all the num_sites sites
# shape is a (N,) vector
R = rlevy(n=N,m=delta,s=gamma) # m is delta, s is gamma
# Raise to the power of phi
R_phi = pow(base=R,exp=phi)
# plt.plot(id_t,R_phi,'.-')
# plt.plot(id_t,R_phi, '.-') # heavy tailed distribution: most observations are small and a few very large

# id8 = np.arange(num_sites)%8 == 0
# plt.plot(id_sites[id8], Z[10,id8],'.-')

# %%
# X_star = R^phi * W
# -----------------------------
# multiply (N, num_site).T with (N,)
# R_phi * W.T is the same as W.T * R_phi
X_star = R_phi * W.T
X_star = X_star.T # shape (N, num_sites)

# Notice now R(s) dominates
# plt.plot(id_t,X_star[:,16], 'b.-')
# plt.plot(id_t,X_star[:,24], 'g.-')
# plt.plot(id_t,X_star[:,199],'r.-')

# %%
# X (with nugget)
# epsilon.shape = (N,num_sites)
epsilon = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(num_sites,)),
                                                cov=tau_square,size=N)
X = X_star + epsilon
np.save('X',X)
# plt.plot(id_t,epsilon[:,0],'.-') # blue
# plt.plot(id_t,X[:,0],'.-') # orange

# %%
# Transform to Y
pool = Pool(processes = n_cores)
start_time = time.time()
results = pool.map(F_X_vec_wrapper,X.ravel())
F_Xs = np.array(results).reshape(X.shape)
end_time = time.time()
print("F_X multicore use: ", round(end_time - start_time, 3), " seconds.")
# F_Xs = F_X_vec(X,phi,gamma,tau) # xval, phi, gamma, tau
pool.close()
pool.join()
np.save("F_Xs",F_Xs)

leq_mask = F_Xs <= p
greater_mask = np.logical_not(leq_mask)

Y = F_Xs.copy()
q = 0
Y[leq_mask] = q
Y[greater_mask] = genpareto.ppf((F_Xs[greater_mask]-p)/(1-p), c=ksi, loc=q, scale=sigma)
np.save("Y",Y)

# %%
# log likelihood single core
# each time use observation from one site across 30 time replicates
# sum up the loglikelihodds (log of a product is the sum of logs)
# log_censored_likelihood_vec = np.vectorize(log_censored_likelihood)

# # a grid of evaluations for phi from 0.05 to 0.95, by 0.05
# phis = np.linspace(start = 0.05, stop = 0.95, num = 19)
# loglik_total = [0]*len(phis)
# for site in np.arange(len(idx)*len(idy)): # loop through all the sites
#     for phi_index in range(len(phis)): # evaluate loglik across phis
#         phi = phis[phi_index]
#         loglik_site_30time = log_censored_likelihood_vec(Y[:,site], u, X_star[:,site], p, phi, gamma, tau, mu, sigma, ksi)
#         loglik_site = sum(loglik_site_30time)
#         loglik_total[phi_index] += loglik_site
# fig, ax = plt.subplots()
# ax.plot(phis, loglik_total, '.-')
# fig.savefig('logliklihood.pdf')

# %%
# log likelihood parallel multi-core
# each time use observation from one site across 30 time replicates
# sum up the loglikelihodds (log of a product is the sum of logs)

sites = np.arange(num_sites)

# Evaluate across phi
# -------------------
phis = np.linspace(start = 0.05, stop = 0.95, num = 19)
# phis = np.linspace(start = 0.2, stop = 0.7, num = 11) # evaluate loglik versus phi
loglik_total = [0]*len(phis) # place to store results

pool = Pool(processes=n_cores)

for phi_index in range(len(phis)):
    phi = phis[phi_index]
    phi_rep = np.repeat(phi,len(sites))
    arg_site_phi = np.vstack((sites,phi_rep)).T
    start_time = time.time()
    loglik_each_site_each_time = pool.starmap(log_censored_likelihood_wrapper,arg_site_phi) # shape (N, num_site)
    loglik_each_site = np.sum(loglik_each_site_each_time, axis = 1) # sum over all N time replicates # shape (1,num_site)
    end_time = time.time()
    print('phi=',str(round(phi,2)),' takes ', round(end_time - start_time, 3), ' seconds')
    np.save('loglik_each_site_at_phi_of_'+str(round(phi,2)), loglik_each_site)
    loglik_at_phi = np.sum(loglik_each_site) # one number
    loglik_total[phi_index] = loglik_at_phi
pool.close()
pool.join()
np.save('loglik_total_across_phi',loglik_total)

fig, ax = plt.subplots()
ax.plot(phis, loglik_total, '.-')
fig.savefig('loglik_across_phi.pdf')

# Evaluate across gamma
# ---------------------
gammas = np.linspace(0.5,1.5,21) # 0.5 to 1.5 by 0.05
loglik_total = [0]*len(gammas)
pool = Pool(processes = n_cores)
for gamma_index in range(len(gammas)):
    gamma = gammas[gamma_index]
    gamma_rep = np.repeat(gamma, len(sites))
    arg_site_gamma = np.vstack((sites,gamma_rep)).T
    loglik_each_site_each_time = pool.starmap(log_censored_likelihood_wrapper_gamma, arg_site_gamma)
    loglik_each_site = np.sum(loglik_each_site_each_time, axis = 1)
    np.save('loglik_each_site_at_gamma_of_'+str(round(gamma,2)), loglik_each_site)
    loglik_at_gamma = np.sum(loglik_each_site)
    loglik_total[gamma_index] = loglik_at_gamma
pool.close()
pool.join()
np.save('loglik_total_across_gamma',loglik_total)
fig, ax = plt.subplots()
ax.plot(gammas, loglik_total, '.-')
fig.savefig('loglik_across_gamma.pdf')

# %%
