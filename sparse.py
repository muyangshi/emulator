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
# 3D
# ------------------------------------------------------------------------
# Train 100x100x100
# Load Data
y_train = np.load('./data/100x100x100/y_train_vector.npy')

p_train = np.linspace(0.8,0.9999,100) # shape (n1,)
phi_train = np.linspace(0,1,100) # shape (n2, )
gamma_train = np.linspace(0.5,2,100) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

# Sparse GP Model
Z_p = np.linspace(0.8,0.9999,10)
Z_phi = np.linspace(1e-10,1,10)
Z_gamma = np.linspace(0.5,2,10)
# Z_train = NDimCoord(Z_phi, Z_gamma)
Z_p_g, Z_phi_g, Z_gamma_g = np.meshgrid(Z_p, Z_phi, Z_gamma, indexing='ij')
Z_train = np.vstack([Z_p_g,Z_phi_g,Z_gamma_g]).reshape(3,-1).T
ker = GPy.kern.RBF(input_dim=3, variance=1, lengthscale=(0.01,0.1,0.1),
                    ARD=True)
m_sparse = GPy.models.SparseGPRegression(X_train,y_train,Z=Z_train,
                                            kernel = ker)
start = time.time()
m_sparse.optimize('bfgs')
print('training used: ', round(time.time() - start, 2), ' seconds.')

os.makedirs('./data/sparse',exist_ok=True)
with open('./data/sparse/m_sparse.pickle', 'wb') as f:
    pickle.dump(m_sparse,f,pickle.HIGHEST_PROTOCOL)

print(m_sparse)

# Evaluate

y_test = np.load('./data/80x80x80/y_test_vector.npy')
p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)

start = time.time()
y_test_pred_sparse, _ = m_sparse.predict(X_test)
np.save('./data/sparse/y_test_pred_sparse',y_test_pred_sparse)
print('prediction used: ', round(time.time() - start, 2), ' seconds.')
MSE(y_test_pred_sparse,y_test)

################################################################################################################################
######################################################    2D Local  ############################################################
################################################################################################################################
# %%
# 2D
# ---------------------------------------------------------------
# Calculate Data
# 100x100
phi_train = np.linspace(1e-10,1,100)
gamma_train = np.linspace(0.5,2,100)
X_train = NDimCoord(phi_train,gamma_train)
phi_train_coor, gamma_train_coor = np.split(X_train,X_train.shape[1],1)
p_train_coor = 0.9
start_time = time.time()
y_train = model_sim.qRW_Newton(p_train_coor, phi_train_coor, gamma_train_coor,100)
print('y calculated. Used: ', round(time.time() - start_time, 2), ' seconds.')

# # 50x50
# phi_train = np.linspace(1e-10,1,50)
# gamma_train = np.linspace(0.5,2,50)
# X_train = NDimCoord(phi_train,gamma_train)
# phi_train_coor, gamma_train_coor = np.split(X_train,X_train.shape[1],1)
# p_train_coor = 0.9
# start_time = time.time()
# y_train = model_sim.qRW_Newton(p_train_coor, phi_train_coor, gamma_train_coor,100)
# print('y calculated. Used: ', round(time.time() - start_time, 2), ' seconds.')

# # 80x80
phi_test = np.linspace(1e-10,1,80)
gamma_test = np.linspace(0.5,2,80)
X_test = NDimCoord(phi_test, gamma_test)
phi_test_coor, gamma_test_coor = np.split(X_test, X_test.shape[1],1)
p_test_coor = 0.9
start_time = time.time()
y_test = model_sim.qRW_Newton(p_test_coor, phi_test_coor, gamma_test_coor, 100)
print('y_test calculated used: ', round(time.time() - start_time, 2), ' seconds.')

# # %%
# full GP model
ker = GPy.kern.RBF(input_dim=2,variance=1,lengthscale=(0.05,0.01),
                    ARD=True)
full_model = GPy.models.GPRegression(X = X_train, Y = y_train,
                                        kernel = ker,
                                        noise_var=1)
                                        # 46 s for 100x100 input
                                        # 1.2s for 50x50 input
full_model.optimize(optimizer='lbfgs')
# # 100 x 100 input will kill the python kernel possibly memory error
# # using 50 x 50 training instead, 1m 10s
# print(full_model)
# print(full_model.kern.lengthscale)

# y_test_pred, _ = full_model.predict(X_test)
# y_train_pred, _ = full_model.predict(X_train)
# MSE(y_test_pred, y_test)
# MSE(y_train_pred, y_train)


# # %%
# Sparse GP Model

Z_phi = np.linspace(1e-10,1,10)
Z_gamma = np.linspace(0.5,2,10)
Z_train = NDimCoord(Z_phi, Z_gamma)
ker = GPy.kern.RBF(input_dim=2, variance=1, lengthscale=(0.05,0.01),
                    ARD=True)
m_sparse = GPy.models.SparseGPRegression(X_train,y_train,Z=Z_train,
                                            kernel = ker)
m_sparse.optimize('bfgs') 
# # (10x5) Z
# # 1m 53s with RBF ARD kernel on 50x50 input
# # 9s with RBF ARD kernel on 100x100 input

# # (10x10) Z
# # 47s with RBF ARD kernel

# print(m_sparse)
y_test_pred_sparse, _ = m_sparse.predict(X_test) 
# # 0.5s (10x5), (10x10)
MSE(y_test_pred_sparse,y_test) 
# # 0.0267 (10x5), 0.0001276 (10x10)

# %%
