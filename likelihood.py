#%%
# Imports
from cmath import exp
from model_sim import *
import numpy as np
import scipy
from scipy.spatial import distance
import matplotlib.pyplot as plt
import math
from scipy.stats import genpareto

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
# Some Math Functions
def norm_to_Pareto(z):
    if(isinstance(z, (int, np.int64, float))): z=np.array([z])
    tmp = scipy.stats.norm.cdf(z)
    if np.any(tmp==1): tmp[tmp==1]=1-1e-9
    return 1/(1-tmp)-1

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
N = 30 # sample size
phi = 0.5 # the phi in R^phi*W
gamma = 1 # is this the gamma that goes in rlevy?
ksi = 0 # Generalized Parato Distribution parameter for Y
mu = 0 # Generalized Parato Distribution parameter for Y
sigma = 1 # Generalized Parato Distribution parameter for Y
tau = np.sqrt(10) # standard deviation of the Gaussian nugget terms
covariance_matrix = 1 # the Cov for multivariate gaussian Z(s)
p = 0.9 # censor likelihood


#%%
# Generate the Z(s) Gaussian Process
id = np.arange(N)
Z = scipy.stats.multivariate_normal.rvs(mean=0,cov=covariance_matrix,size=N,random_state=2)
plt.plot(id,Z,'b.-')


#%%
# Transform Z(s) to g(Z(s)) = W(s)
W = norm_to_Pareto(Z)
# plt.ylim(top=10)
plt.plot(id,W,'r.-')
# plt.show()

# %%
# Generate R(s) Levy
R = rlevy(n=N)
plt.plot(id,R,'g.-')

# %%
# Raise to the power of phi
R_phi = pow(base=R,exp=0.5)
plt.plot(id,R_phi,'.-')

# %%
# X_star
X_star = R_phi*W
plt.plot(id,X_star,'.-')

# %%
# X (with nugget)
epsilon = scipy.stats.norm.rvs(loc=0, scale=tau, size=N, random_state=3)
plt.plot(id,epsilon,'.-') # blue
X = X_star + epsilon
plt.plot(id,X,'.-') # orange

# %%
# Transform to Y
F_X_vec = np.vectorize(F_X)
F_Xs = F_X_vec(X,0.5,1,10) # xval, phi, gamma, tau
Y = genpareto.ppf(F_Xs,c=0)
plt.plot(id,Y,'.-')
# u = genpareto.ppf(0.8,c=0)

# %%
# Calculate log-likelihood across a range of phi's
u = genpareto.ppf(0.8,c=0)
log_censored_likelihood_vec = np.vectorize(log_censored_likelihood)
phis = np.linspace(start = 0.05, stop = 0.95, num=10)
loglik = []
for phi in phis:
    logliks = log_censored_likelihood_vec(Y, u, X_star, 0.8, phi, gamma, tau, mu, sigma, ksi)
    loglik.append(sum(logliks))
plt.plot(phis,loglik,'.-')

# %%
# Calculate log-likelihood across a range of phi's
u = genpareto.ppf(0.9,c=0)
log_censored_likelihood_vec = np.vectorize(log_censored_likelihood)
phis = np.linspace(start = 0.05, stop = 0.95, num=10)
loglik = []
for phi in phis:
    logliks = log_censored_likelihood_vec(Y, u, X_star, 0.9, phi, gamma, tau, mu, sigma, ksi)
    loglik.append(sum(logliks))
plt.plot(phis,loglik,'.-')
# %%
