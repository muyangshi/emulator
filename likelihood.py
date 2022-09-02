#%%
# imports
from model_sim import *
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.stats import genpareto
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import pairwise_distances

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

def rlevy(n, m = 0, s = 1): # m is delta, s is gamma in Stable(alpha, 1, gamma, delta)
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

# def cov_spatial(r, cov_model = "exponential", cov_pars = np.array([1,1]), kappa = 0.5):
#     if type(r).__module__!='numpy' or isinstance(r, np.float64):
#       r = np.array(r)
#     if np.any(r<0):
#       sys.exit('Distance argument must be nonnegative.')
#     r[r == 0] = 1e-10
    
#     if cov_model != "matern" and cov_model != "gaussian" and cov_model != "exponential" :
#         sys.exit("Please specify a valid covariance model (matern, gaussian, or exponential).")
    
#     if cov_model == "exponential":
#         C = np.exp(-r)
    
#     if cov_model == "gaussian" :
#         C = np.exp(-(r^2))
  
#     if cov_model == "matern" :
#         range = 1
#         nu = kappa
#         part1 = 2 ** (1 - nu) / sc.gamma(nu)
#         part2 = (r / range) ** nu
#         part3 = sc.kv(nu, r / range)
#         C = part1 * part2 * part3
#     return C

# def ns_cov(range_vec, sigsq_vec, coords, kappa = 0.5, cov_model = "matern"):
#     if type(range_vec).__module__!='numpy' or isinstance(range_vec, np.float64):
#       range_vec = np.array(range_vec)
#       sigsq_vec = np.array(sigsq_vec)
    
#     N = range_vec.shape[0] # Number of spatial locations
#     if coords.shape[0]!=N: 
#       sys.exit('Number of spatial locations should be equal to the number of range parameters.')
  
#     # Scale matrix
#     arg11 = range_vec
#     arg22 = range_vec
#     arg12 = np.repeat(0,N)
#     ones = np.repeat(1,N)
#     det1  = arg11*arg22 - arg12**2
  
#     ## --- Outer product: matrix(arg11, nrow = N) %x% matrix(1, ncol = N) --- 
#     mat11_1 = np.reshape(arg11, (N, 1)) * ones
#     ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg11, ncol = N) ---
#     mat11_2 = np.reshape(ones, (N, 1)) * arg11
#     ## --- Outer product: matrix(arg22, nrow = N) %x% matrix(1, ncol = N) ---
#     mat22_1 = np.reshape(arg22, (N, 1)) * ones  
#     ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg22, ncol = N) ---
#     mat22_2 = np.reshape(ones, (N, 1)) * arg22
#     ## --- Outer product: matrix(arg12, nrow = N) %x% matrix(1, ncol = N) ---
#     mat12_1 = np.reshape(arg12, (N, 1)) * ones 
#     ## --- Outer product: matrix(1, nrow = N) %x% matrix(arg12, ncol = N) ---
#     mat12_2 = np.reshape(ones, (N, 1)) * arg12
  
#     mat11 = 0.5*(mat11_1 + mat11_2)
#     mat22 = 0.5*(mat22_1 + mat22_2)
#     mat12 = 0.5*(mat12_1 + mat12_2)
  
#     det12 = mat11*mat22 - mat12**2
  
#     Scale_mat = np.diag(det1**(1/4)).dot(np.sqrt(1/det12)).dot(np.diag(det1**(1/4)))
  
#     # Distance matrix
#     inv11 = mat22/det12
#     inv22 = mat11/det12
#     inv12 = -mat12/det12
  
#     dists1 = distance.squareform(distance.pdist(np.reshape(coords[:,0], (N, 1))))
#     dists2 = distance.squareform(distance.pdist(np.reshape(coords[:,1], (N, 1))))
  
#     temp1_1 = np.reshape(coords[:,0], (N, 1)) * ones
#     temp1_2 = np.reshape(ones, (N, 1)) * coords[:,0]
#     temp2_1 = np.reshape(coords[:,1], (N, 1)) * ones
#     temp2_2 = np.reshape(ones, (N, 1)) * coords[:,1]
  
#     sgn_mat1 = ( temp1_1 - temp1_2 >= 0 )
#     sgn_mat1[~sgn_mat1] = -1
#     sgn_mat2 = ( temp2_1 - temp2_2 >= 0 )
#     sgn_mat2[~sgn_mat2] = -1
  
#     dists1_sq = dists1**2
#     dists2_sq = dists2**2
#     dists12 = sgn_mat1*dists1*sgn_mat2*dists2
  
#     Dist_mat_sqd = inv11*dists1_sq + 2*inv12*dists12 + inv22*dists2_sq
#     Dist_mat = np.zeros(Dist_mat_sqd.shape)
#     Dist_mat[Dist_mat_sqd>0] = np.sqrt(Dist_mat_sqd[Dist_mat_sqd>0])
  
#     # Combine
#     Unscl_corr = cov_spatial(Dist_mat, cov_model = cov_model, cov_pars = np.array([1,1]), kappa = kappa)
#     NS_corr = Scale_mat*Unscl_corr
  
#     Spatial_cov = np.diag(sigsq_vec).dot(NS_corr).dot(np.diag(sigsq_vec)) 
#     return(Spatial_cov)

# np.vectorize(censored_likelihood)

## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##
##          Simulate Dataset Check Smoothness of Censored Likelihood
## -------------------------------------------------------------------------- ##
## -------------------------------------------------------------------------- ##

# %%
# Simulation Parameters
N = 30 # number of time replicates
phi = 0.5 # the phi in R^phi*W
gamma = 1 # is this the gamma that goes in rlevy?
ksi = 0 # Generalized Parato Distribution parameter for Y
mu = 0 # Generalized Parato Distribution parameter for Y
sigma = 1 # Generalized Parato Distribution parameter for Y
tau = np.sqrt(10) # standard deviation of the Gaussian nugget terms
covariance_matrix = 1 # the Cov for multivariate gaussian Z(s)
p = 0.9 # censoring threshold probability
u = genpareto.ppf(p,c=0, loc=mu, scale=sigma) # censoring threshold value
rho = 0.2
length_scale = 1/rho # scikit/learn parameterization (length_scale)
nu = 0.5 # exponential kernel


# %% 
# Generate Z(s) Gaussian Process, 25 locations, N time replicates
idx = np.linspace(1,5,num=5) # x locations [1,2,3,4,5]
idy = np.linspace(1,5,num=5) # y locations [1,2,3,4,5]
id = np.arange(N)
# 25 locations in a 2D array
s_xy = np.vstack(np.meshgrid(idx,idy,indexing='ij')).reshape(2,-1).T
# 25 by 25 euclidean distance matrix, entry (i,j) is h(point_i,point_j)
distance_matrix = pairwise_distances(s_xy, metric="euclidean")
# covariance kernel, with nu=0.5, it's exp(-rho*x)
K = Matern(nu=nu,length_scale=length_scale)(0, distance_matrix.ravel()[:,np.newaxis]).reshape(distance_matrix.shape)
# Generate N time replicates, Z.shape = (30,25) --> 30 time replicates for 25 locations
# each row is a time replicate of 25 sites
# each column is one site, across N time replicates
Z = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(25,)),cov=K,size=N,random_state=2)
# Plot the first location
# plt.plot(id,Z[:,0],'.-')
# plt.plot(id,Z[:,1],'r.-')
# plt.plot(id,Z[:,6],'b.-')
# plt.plot(id,Z[:,24],'g.-')
# d = pairwise_distances([[1,0],[3,0],[5,0]], metric="euclidean")
# K = Matern(nu=0.5, length_scale=1/0.2)(0, d.ravel()[:, np.newaxis]).reshape(d.shape)

# %%
# Transform Z(s) to W(s) = g(Z(s))
W = norm_to_Pareto(Z) # shape = (30,25) --> 30 time replicates for 25 locations
# plt.plot(id,W[:,0],'.-')
# plt.plot(id,W[:,1],'r.-')
# plt.plot(id,W[:,6],'b.-')
# plt.plot(id,W[:,24],'g.-')

# %%
# Generate R(s) Levy
# Generate one R for all the sites at each time replicate, for 30 replicates
# shape is thus a (30,) vector
R = rlevy(n=30,m=0,s=1) # s is gamma, m is delta
# Raise to the power of phi
R_phi = pow(base=R,exp=phi)
# plt.plot(id,R_phi,'.-')
plt.plot(id,R_phi, '.-') # heavy tailed distribution: most observations are small and a few very large

# %%
# Take to product and yield X_star
# Notice now R(s) dominates
X_star = R_phi * W.T # multiply (30,) with (30,25)*T; R_phi * W.T is the same as W.T * R_phi
X_star = X_star.T
plt.plot(id,X_star[:,0],'.-')
# plt.plot(id,X_star[1],".-")

# %%
# X (with nugget)
# epsilon.shape = (30,25), 30 replicates for 25 locations
epsilon = scipy.stats.multivariate_normal.rvs(mean=np.zeros(shape=(25,)),cov=10,size=N,random_state=53)
plt.plot(id,epsilon[:,0],'.-') # blue
X = X_star + epsilon
plt.plot(id,X[:,0],'.-') # orange

# %%
# Transform to Y
F_X_vec = np.vectorize(F_X)
F_Xs = F_X_vec(X,phi,gamma,tau) # xval, phi, gamma, tau # shape (30,25) # takes 20 secs

leq_mask = F_Xs <= 0.9
greater_mask = np.logical_not(leq_mask)

Y = F_Xs.copy()
Y[leq_mask] = genpareto.ppf(F_Xs[leq_mask], c=ksi)
Y[greater_mask] = genpareto.ppf((F_Xs[greater_mask]-p)/(1-p), c=ksi)

plt.plot(id,Y[:,0],'.-')

# %%
# log likelihood
# each time use observation from one site across 30 time replicates
# sum up the loglikelihodds (log of a product is the sum of logs)
log_censored_likelihood_vec = np.vectorize(log_censored_likelihood)
phis = np.linspace(start = 0.2, stop = 0.7, num = 11)
loglik_total = [0]*len(phis)
for site in np.arange(len(idx)*len(idy)): # loop through all the sites
    for phi_index in range(len(phis)): # evaluate loglik across phis
        phi = phis[phi_index]
        loglik_site_30time = log_censored_likelihood_vec(Y[:,site], u, X_star[:,site], p, phi, gamma, tau, mu, sigma, ksi)
        loglik_site = sum(loglik_site_30time)
        loglik_total[phi_index] += loglik_site
fig, ax = plt.subplots()
ax.plot(phis, loglik_total, '.-')
fig.savefig('logliklihood.pdf')