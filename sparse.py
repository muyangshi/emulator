# %%
# imports
# -------
import math
import time
import os
import pickle
from operator import itemgetter
from distutils.util import strtobool

import numpy as np
import matplotlib.pyplot as plt

# import gp_emulator
import model_sim
import GPy
np.random.seed(42)

# %%
# Define Helper Functions
# -----------------------

def NDimCoord(*args):
    """
    Converts n 1-Dim vectors (of length v1,v2,...,vn) to n-Dim grid
    then, return the coordinates, which is ndarray of shape (v1*v2*...*vn,n)

    Usage Example
    x = np.linspace(0,1,100)
    y = np.linspace(0,1,100)
    grid = NDimCoord(x,y)
    """
    return np.vstack(np.meshgrid(*args)).reshape((len(args),-1)).T

def SSE(y_pred, y_true):
    return sum((y_pred - y_true)**2)

def MSE(y_pred, y_true):
    return SSE(y_pred, y_true)/y_true.size

def split_data(X,y,training_proportion,path):
    # Partition training and testing set
    np.random.seed(42)
    n_samples = y.size
    training_size = math.floor(n_samples * training_proportion)
    training_indices = np.random.choice(n_samples,training_size,replace = False)
    testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
    X_train, y_train = X[training_indices], y[training_indices]
    X_test, y_test = X[testing_indices], y[testing_indices]
    # Save the partitioned training and testing dataset
    np.save(path + 'X_train', X_train)
    np.save(path + 'X_test', X_test)
    np.save(path + 'y_train', y_train)
    np.save(path + 'y_test', y_test)
    return X_train, X_test, y_train, y_test

def draw3D(quantile,X,Y,Z_pred,Z_true):
    """
    quantile is a string (also used as title of plot)
    X, Y, Z_pred, and Z_true are 1D arrays, shape (n,)
    """
    fig = plt.figure()
    ax = plt.axes(projection = '3d')
    ax.plot_trisurf(X,Y,Z_pred,color='orange') # Predicted
    ax.plot_trisurf(X,Y,Z_true,color='blue') # True
    ax.set_title(quantile)
    ax.set_xlabel(r"$\phi$")
    ax.set_ylabel(r'$\gamma$')
    ax.set_zlabel(r'$F^{-1}$')
    plt.show()

# %%
# Calculate Data
# # 100x100
# phi_train = np.linspace(1e-10,1,100)
# gamma_train = np.linspace(0.5,2,100)
# X_train = NDimCoord(phi_train,gamma_train)
# phi_train_coor, gamma_train_coor = np.split(X_train,X_train.shape[1],1)
# p_train_coor = 0.9
# start_time = time.time()
# y_train = model_sim.qRW_Newton(p_train_coor, phi_train_coor, gamma_train_coor,100)
# print('y calculated. Used: ', round(time.time() - start_time, 2), ' seconds.')

# 50x50
phi_train = np.linspace(1e-10,1,50)
gamma_train = np.linspace(0.5,2,50)
X_train = NDimCoord(phi_train,gamma_train)
phi_train_coor, gamma_train_coor = np.split(X_train,X_train.shape[1],1)
p_train_coor = 0.9
start_time = time.time()
y_train = model_sim.qRW_Newton(p_train_coor, phi_train_coor, gamma_train_coor,100)
print('y calculated. Used: ', round(time.time() - start_time, 2), ' seconds.')

# 80x80
phi_test = np.linspace(1e-10,1,80)
gamma_test = np.linspace(0.5,2,80)
X_test = NDimCoord(phi_test, gamma_test)
phi_test_coor, gamma_test_coor = np.split(X_test, X_test.shape[1],1)
p_test_coor = 0.9
start_time = time.time()
y_test = model_sim.qRW_Newton(p_test_coor, phi_test_coor, gamma_test_coor, 100)
print('y_test calculated used: ', round(time.time() - start_time, 2), ' seconds.')

# %%
# full GP model
ker = GPy.kern.RBF(input_dim=2,variance=1,lengthscale=(0.05,0.01),
                    ARD=True)
full_model = GPy.models.GPRegression(X = X_train, Y = y_train,
                                        kernel = ker,
                                        noise_var=1)
                                        # 46 s for 100x100 input
                                        # 1.2s for 50x50 input
full_model.optimize(optimizer='lbfgs')
# 100 x 100 input will kill the python kernel possibly
# using 50 x 50 training instead, 1m 10s
print(full_model)
print(full_model.kern.lengthscale)

y_test_pred, _ = full_model.predict(X_test)
y_train_pred, _ = full_model.predict(X_train)
MSE(y_test_pred, y_test)
MSE(y_train_pred, y_train)


# %%
# Sparse GP Model

Z_phi = np.linspace(1e-10,1,10)
Z_gamma = np.linspace(0.5,2,10)
Z_train = NDimCoord(Z_phi, Z_gamma)
ker = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=(0.05,0.01),
                    ARD=True)
m_sparse = GPy.models.SparseGPRegression(X_train,y_train,Z=Z_train,
                                            kernel = ker)
m_sparse.optimize('bfgs') 
# (10x5)
# 1m 53s with RBF ARD kernel on 50x50
# 9s with RBF ARD kernel on 100x100

# (10x10)
# 47s with RBF ARD kernel

print(m_sparse)
y_test_pred_sparse, _ = m_sparse.predict(X_test) 
# 0.5s (10x5), (10x10)
MSE(y_test_pred_sparse,y_test) 
# 0.0267 (10x5), 0.0001276 (10x10)




# %%
# # 0.5 422 seconds (100x100)
# # 0.5 ~20 seconds (50x50)
# m_full = GPy.models.GPRegression(X_train,y_train)
# start = time.time()
# m_full.optimize('lbfgs')
# print('full: ', time.time() - start)
# # %%
# start = time.time()
# y_pred,_ = m_full.predict(X)
# print('full pred: ', time.time() - start)
# MSE(y_pred, y) # 1.xx e -9
# draw3D('full', phi_coor.ravel(), gamma_coor.ravel(), y_pred.ravel(), y.ravel())


# # 33 seconds ??
# Z_phi = np.linspace(1e-10,1,5)
# Z_gamma = np.linspace(0.5,2,5)
# Z = NDimCoord(Z_phi,Z_gamma)
# m = GPy.models.SparseGPRegression(X_train,y_train,Z=Z)
# start = time.time()
# m.optimize('bfgs')
# print('sparse: ', time.time() - start)
# y_pred_sparse, _ = m.predict(X)
# draw3D('sparse', phi_coor.ravel(), gamma_coor.ravel(),y_pred_sparse.ravel(),y.ravel())
# MSE(y_pred_sparse, y)
# %%
