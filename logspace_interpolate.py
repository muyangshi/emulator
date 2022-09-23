# %%
# imports
# -----------------------------------------------------------------
from model_sim import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

from multiprocessing import Pool
import os
import time
import pickle
# %%
# Utilities
F_X_star = lib.pmixture_C
quantile_F_X_star = lib.qRW_newton_C

F_X = lib.F_X_cheat
quantile_F_X = lib.quantile_F_X

f_X = lib.f_X

def quantile_F_X_star_wrapper(p, phi, gamma):
    return quantile_F_X_star(p,phi,gamma,100)

def quantile_F_X_wrapper(p, phi, gamma, tau):
    return quantile_F_X(p, phi, gamma, tau)

"""
If otypes is not specified, then a call to the function with the first argument will be used 
to determine the number of outputs. 
The results of this call will be cached 
if cache is True to prevent calling the function twice. 
However, to implement the cache, 
the original function must be wrapped which will slow down subsequent calls, 
so only do this if your function is expensive.
"""
quantile_F_X_vec = np.vectorize(quantile_F_X,otypes=[float])
quantile_F_X_star_vec = np.vectorize(quantile_F_X_star, otypes=[float])

# MSE
def SSE(y_pred, y_true):
    return sum((y_pred - y_true)**2)
def MSE(y_pred, y_true):
    return SSE(y_pred, y_true)/len(y_true)
# Relative (absolute) Errors
def absolute_relative_error(y_pred, y_true):
    """
    returns the average of absolute relative errors
    """
    abs_rel_res = abs(y_true - y_pred)/y_true
    return sum(abs_rel_res)/len(abs_rel_res)

# %%

############################
##    p, phi, gamma       ##
############################

# # %%
# # Construct Grid (p, phi, gamma)
# # ------------------------------
n_cores = 50
n_train = 200

# # linear spaced training grids
# # ----------------------------
# p_train = np.linspace(0.8,0.9999,n_train)
# phi_train = np.linspace(0,1,n_train)
# gamma_train = np.linspace(0.5,2,n_train)
# p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
# X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
# start = time.time()
# pool = Pool(processes=n_cores)
# results = pool.starmap(quantile_F_X_star_wrapper,X_train)
# y_train_matrix = np.array(results).reshape(n_train,n_train,n_train)
# # print('Calculation of y train took: ', round((time.time() - start)/60,2), ' minutes.')
# y_train_matrix_ravel = y_train_matrix.ravel()
# path = './data/linspace/200x200x200/'
# os.makedirs(path, exist_ok = True)
# np.save(path+'y_train_matrix',y_train_matrix)
# np.save(path+'y_train_matrix_ravel',y_train_matrix_ravel)


# # log spaced training grids
# # -------------------------
# p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,n_train)
# phi_train = 2 + np.geomspace(-2,-1,n_train)
# gamma_train = (2 + 0.5) + np.geomspace(-2,-0.5,n_train)

# p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
# X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
# X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

# start = time.time()
# pool = Pool(processes=n_cores)
# results = pool.starmap(quantile_F_X_star_wrapper,X_train)
# y_train_matrix = np.array(results).reshape(n_train,n_train,n_train)
# print('Calculation of y train took: ', round((time.time() - start)/60,2), ' minutes.')
# y_train_matrix_ravel = y_train_matrix.ravel()
# # y_train_vector = qRW_Newton(X_train_p, X_train_phi, X_train_gamma, 100)

# path = './data/logspace/200x200x200/'
# os.makedirs(path, exist_ok = True)
# np.save(path+'y_train_matrix',y_train_matrix)
# np.save(path+'y_train_matrix_ravel',y_train_matrix_ravel)
# # np.save(path+'y_train_vector',y_train_vector) # y_train_vector.ravel() = y_train_matrix_ravel, using indexing='ij'

# Mix spaced training grids
# -------------------------
# p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,n_train)
# phi_train = np.linspace(0,1,n_train)
# gamma_train = np.linspace(0.5,2,n_train)
# p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
# X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
# start = time.time()
# pool = Pool(processes=n_cores)
# results = pool.starmap(quantile_F_X_star_wrapper,X_train)
# y_train_matrix = np.array(results).reshape(n_train,n_train,n_train)
# print('Calculation of y train mixspace took: ', round((time.time() - start)/60,2), ' minutes.')
# path = './data/mixspace/200x200x200/'
# os.makedirs(path, exist_ok = True)
# np.save(path+'y_train_matrix',y_train_matrix)


# # linear spaced testing grids 80x80x80
# # ------------------------------------
# n_test = 80

# p_test = np.linspace(0.8,0.9999,80)
# phi_test = np.linspace(0,1,80) # shape(m1,)
# gamma_test = np.linspace(0.5,2,80) # shape(m2,)

# p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
# X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
# X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

# pool = Pool(processes=n_cores)
# results = pool.starmap(quantile_F_X_star_wrapper, X_test)
# y_test_matrix = np.array(results).reshape(n_test,n_test,n_test)
# # y_test_matrix = qRW_Newton(p_test_g,phi_test_g, gamma_test_g, 100)
# y_test_matrix_ravel = y_test_matrix.ravel()
# # y_test_vector = qRW_Newton(X_test_p,X_test_phi,X_test_gamma,100)

# path = './data/linspace/80x80x80/'
# os.makedirs(path, exist_ok=True)
# np.save(path+'y_test_matrix',y_test_matrix)
# np.save(path+'y_test_matrix_ravel',y_test_matrix_ravel)
# # np.save(path+'y_test_vector',y_test_vector)

# %%
# Construct Interpolator
# ----------------------
n_train = 200

# # log spaced training grids
# # -------------------------
# p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,n_train)
# phi_train = 2 + np.geomspace(-2,-1,n_train)
# gamma_train = (2 + 0.5) + np.geomspace(-2,-0.5,n_train)
# y_train_matrix = np.load('./data/logspace/200x200x200/y_train_matrix.npy')
# interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
# with open('./data/logspace/200x200x200/interpolate.pickle','wb') as f:
#     pickle.dump(interpolate3D, f, pickle.HIGHEST_PROTOCOL)

# # linear spaced training grids
# # ----------------------------
# p_train = np.linspace(0.8,0.9999,n_train)
# phi_train = np.linspace(0,1,n_train)
# gamma_train = np.linspace(0.5,2,n_train)
# y_train_matrix = np.load('./data/linspace/200x200x200/y_train_matrix.npy')
# interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
# with open('./data/linspace/200x200x200/interpolate.pickle','wb') as f:
#     pickle.dump(interpolate3D, f, pickle.HIGHEST_PROTOCOL)

# mix spaced training grids
# -------------------------
# p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,n_train)
# phi_train = np.linspace(0,1,n_train)
# gamma_train = np.linspace(0.5,2,n_train)
# y_train_matrix = np.load('./data/mixspace/200x200x200/y_train_matrix.npy')
# interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
# with open('./data/mixspace/200x200x200/interpolate.pickle','wb') as f:
    # pickle.dump(interpolate3D, f, pickle.HIGHEST_PROTOCOL)

# %%
# Load and Benchmark logspace grid linear interpolator
# ----------------------------------------------------

# Preparations

# n_test = 80
# p_test = np.linspace(0.8,0.9999,80)
# phi_test = np.linspace(0,1,80) # shape(m1,)
# gamma_test = np.linspace(0.5,2,80) # shape(m2,)
# p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
# X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
# y_test_matrix_ravel = np.load('./data/linspace/80x80x80/y_test_matrix_ravel.npy')

# logspace model benchmarks
# -------------------------
# with open('./data/logspace/200x200x200/interpolate.pickle','rb') as f:
#     interpolate3D = pickle.load(f)
# y_pred = interpolate3D(X_test)
# mse = MSE(y_pred, y_test_matrix_ravel)
# print('MSE: ', mse)
# rel_err = []
# for p in p_test:
#     idx = np.where(X_test[:,0] == p)
#     rel_err.append(absolute_relative_error(y_pred[idx], y_test_matrix_ravel[idx]))
# fig, ax = plt.subplots()
# ax.plot(p_test, rel_err, marker='o')
# fig.savefig('./data/logspace/200x200x200/absolute_relative_error.pdf')
# print('p_test: ', p_test)
# print('rel_err: ', rel_err)
# print('mean rel err: ', sum(rel_err)/len(p_test))

# mixspace model benchmarks
# -------------------------
# print("mixspace model")
# with open('./data/mixspace/200x200x200/interpolate.pickle','rb') as f:
#     interpolate3D = pickle.load(f)
# y_pred = interpolate3D(X_test)
# mse = MSE(y_pred, y_test_matrix_ravel)
# print('MSE: ', mse)
# rel_err = []
# for p in p_test:
#     idx = np.where(X_test[:,0] == p)
#     rel_err.append(absolute_relative_error(y_pred[idx], y_test_matrix_ravel[idx]))
# print('mean rel err: ', sum(rel_err)/len(p_test))


# compare logspace and linspace and mixspace model
# models = ['./data/linspace/200x200x200/interpolate.pickle', 
#             './data/logspace/200x200x200/interpolate.pickle',
#             './data/mixspace/200x200x200/interpolate.pickle']
# fig, ax = plt.subplots()
# for model in models:
#     with open(model,'rb') as f:
#         interpolate3D = pickle.load(f)
#     y_pred = interpolate3D(X_test)
#     rel_err = []
#     for p in p_test:
#         idx = np.where(X_test[:,0] == p)
#         rel_err.append(absolute_relative_error(y_pred[idx] , y_test_matrix_ravel[idx]))
#     ax.plot(p_test, rel_err, marker = '.', label = model, alpha=0.5)

# plt.ylim(0,0.01)
# ax.set_ylabel('avg rel err (across phi and gamma) in prop.')
# ax.set_xlabel('p')
# legend = ax.legend(loc='upper center', shadow=True)
# fig.savefig('log-lin-mix-comparison.pdf')

# %%





















###################################
#######    p, phi, gamma, tau #####
###################################

# %%
# Construct Grid (p, phi, gamma, tau)
# -----------------------------------
n_cores = 50
n_train = 4

# Mix spaced training grids 200x200x4x50
# -------------------------
print('Mix spaced training grids 200x200x4x50')

p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,n_train)
phi_train = np.linspace(0,1,n_train)
gamma_train = np.linspace(0.5,2,4)
tau_train = np.linspace(0.1,500,num=4)
p_train_g, phi_train_g, gamma_train_g, tau_train_g = np.meshgrid(p_train, phi_train, gamma_train, tau_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g, tau_train_g]).reshape(4,-1).T
X_train_p, X_train_phi, X_train_gamma, X_train_tau = np.split(X_train,4,-1)

# y_train_vector = quantile_F_X_vec(X_train_p, X_train_phi, X_train_gamma, X_train_tau)

start = time.time()
pool = Pool(processes=n_cores)
results = pool.starmap(quantile_F_X_wrapper,X_train)
y_train_matrix = np.array(results).reshape(n_train,n_train,4,50)
print('Calculation of y train mixspace took: ', round((time.time() - start)/60,2), ' minutes.')
path = './data/mixspace/200x200x4x50/'
os.makedirs(path, exist_ok = True)
np.save(path+'y_train_matrix',y_train_matrix)

pool.close()
pool.join()

# quantile_F_X_vec = np.vectorize(quantile_F_X)
# y_train_vector = quantile_F_X_vec(X_train_p, X_train_phi, X_train_gamma, X_train_tau)
# np.array_equal(y_train_matrix, y_train_vector.reshape(n_train, n_train, n_train, 5)) # True

# Linear Spaced Training Grids 80x80x8x40
# ----------------------------
print('Linear Spaced Training Grids 80x80x8x40')

n_test = 80

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,8) # shape(m2,)
tau_test = np.linspace(0.1,500, num=40)
p_test_g, phi_test_g, gamma_test_g, tau_test_g = np.meshgrid(p_train, phi_train, gamma_train, tau_train, indexing='ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g, tau_test_g]).reshape(4,-1).T
X_test_p, X_test_phi, X_test_gamma, X_test_tau = np.split(X_train,4,-1)

pool = Pool(processes=n_cores)
results = pool.starmap(quantile_F_X_wrapper, X_test)
y_test_matrix = np.array(results).reshape(n_test,n_test,8,40)
y_test_matrix_ravel = y_test_matrix.ravel()

pool.close()
pool.join()

path = './data/linspace/80x80x8x40/'
os.makedirs(path, exist_ok=True)
np.save(path+'y_test_matrix',y_test_matrix)
np.save(path+'y_test_matrix_ravel',y_test_matrix_ravel)

# %%
# Construct Interpolator
# ----------------------

# Mix space training grids
# ------------------------
n_train = 200
p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,n_train)
phi_train = np.linspace(0,1,n_train)
gamma_train = np.linspace(0.5,2,4)
tau_train = np.linspace(0.1,500,num=50)
y_train_matrix = np.load('./data/mixspace/200x200x4x50/y_train_matrix.npy')
interpolate4D = RegularGridInterpolator((p_train, phi_train, gamma_train, tau_train), y_train_matrix)
with open('./data/mixspace/200x200x4x50/interpolate.pickle','wb') as f:
    pickle.dump(interpolate4D, f, pickle.HIGHEST_PROTOCOL)

# %%
# Load and Benchmark logspace grid linear interpolator 4D
# ----------------------------------------------------

# Preparations

n_test = 80

p_test = np.linspace(0.8,0.9999,n_test)
phi_test = np.linspace(0,1,n_test) # shape(m1,)
gamma_test = np.linspace(0.5,2,8) # shape(m2,)
tau_test = np.linspace(0.1,500, num=40)
p_test_g, phi_test_g, gamma_test_g, tau_test_g = np.meshgrid(p_train, phi_train, gamma_train, tau_train, indexing='ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g, tau_test_g]).reshape(4,-1).T
y_test_matrix_ravel = np.load('./data/linspace/80x80x8x40/y_test_matrix_ravel.npy')

with open('./data/mixspace/200x200x4x50/interpolate.pickle','rb') as f:
    interpolate4D = pickle.load(f)
start_time = time.time()
y_pred = interpolate4D(X_test)
end_time = time.time()
print('Calculation of y_pred 80x80x8x40 took: ', round((end_time - start_time)/60,2), ' minutes.')