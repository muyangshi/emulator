# %%
# imports
# -----------------------------------------------------------------
from model_sim import *

import time
import pickle
import os
import gc

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern

import scipy
from scipy.interpolate import RegularGridInterpolator
print('scipy version: ', scipy.__version__)

# Use Ctrl+Shift+P ==> Python interpreter ==> Global (/bin/python3)

# %%
# define helper functions
# -----------------------------------------------------------------
def SSE(y_pred, y_true):
    return sum((y_pred - y_true)**2)

def MSE(y_pred, y_true):
    return SSE(y_pred, y_true)/y_true.size

def abs_rel_err(y_pred, y_true):
    abs_rel_res = abs(y_true - y_pred)/y_true
    return sum(abs_rel_res)/len(abs_rel_res)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
# p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
# X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test = np.vstack(np.meshgrid(p_test,phi_test,gamma_test,indexing='ij')).reshape(3,-1).T
y_test_matrix_ravel = np.load('./remote_data/80x80x80_C/'+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )

# %%
# Construct Interpolator
# ---------------------------------------------------------------------------------------
p_train = np.append(np.linspace(0.8,0.9,100),np.linspace(0.9,0.99999,401)) # shape (n1,)
p_train = np.delete(p_train, 100) # remove the duplicate
phi_train = np.append(np.linspace(0,0.6,100),np.linspace(0.6,1,401)) # shape (n2, )
phi_train = np.delete(phi_train,100) # remove the duplicate
gamma_train = np.append(np.linspace(0.5,1.5,100), np.linspace(1.5,2,401)) # shape (n3, )
gamma_train = np.delete(gamma_train,100) # remove the duplicate
# p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
# X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train = np.vstack(np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')).reshape(3,-1).T
y_train_matrix = np.load('./remote_data/100_400_3/y_train_matrix.npy') # shape (n1, n2, n3)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
os.makedirs('./data/linear/',exist_ok=True)
with open('./data/linear/interpolate3D_100_400_3.pickle','wb') as f:
    pickle.dump(interpolate3D,f,pickle.HIGHEST_PROTOCOL)

# %%
# Time Evaluation
# -----------------------------------------------------------------------------------------------
models = ['interpolate3D_100.pickle',
            'interpolate3D_200.pickle',
            'interpolate3D_400.pickle',
            'interpolate3D_500.pickle',
            'interpolate3D_100_300_2.pickle',
            'interpolate3D_100_400_2.pickle']
for model in models:
    print(model)
    with open('./data/models/'+model,'rb') as f:
        interpolate3D = pickle.load(f)
    n_points = [10,50,100,200,300]
    for n in n_points:
        print(n,"**3")
        p_test = np.linspace(0.8,0.9999,n)
        phi_test = np.linspace(0,1,n) # shape(m1,)
        gamma_test = np.linspace(0.5,2,n) # shape(m2,)
        p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
        X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
        start = time.time()
        _ = interpolate3D(X_test)
        print(time.time() - start)

# %%
# relative (absolute) errors by p levels
# ------------------------------------------------------------------------------------------------
fig, ax = plt.subplots()
models = ['interpolate3D_100_400_3.pickle']
models = ['interpolate3D_100.pickle',
            'interpolate3D_200.pickle',
            'interpolate3D_400.pickle',
            'interpolate3D_500.pickle',
            'interpolate3D_100_300_2.pickle',
            'interpolate3D_100_400_2.pickle',
            'interpolate3D_100_400_3.pickle']
# models = ['interpolate3D_500.pickle',
#             'interpolate3D_100_300_2.pickle',
#             'interpolate3D_100_400_2.pickle']
for model in models:
    with open('./data/linear/'+model,'rb') as f:
        interpolate3D = pickle.load(f)
    y_pred = interpolate3D(X_test)
    rel_err = []
    # MSEs = []
    for p in p_test:
        idx = np.where(X_test[:,0] == p)
        rel_err.append(abs_rel_err(y_pred[idx], y_test_matrix_ravel[idx]))
    # MSEs.append(MSE(y_pred[idx],y_test_matrix_ravel[idx]))
    ax.plot(p_test, rel_err, marker='o',label=model)

ax.set_ylabel('in %')
ax.set_xlabel('p')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.xlim([0.95,0.99])
plt.ylim([0,1])
plt.show()


# %%
# splines scipy 1.9.0
# ---------------------------------------------------------
p_train = np.linspace(0.8,0.9999,100) # shape (n1,)
phi_train = np.linspace(0,1,100) # shape (n2, )
gamma_train = np.linspace(0.5,2,100) # shape (n3, )
X_train = np.vstack(np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')).reshape(3,-1).T # shape(n1*n2*n3,3)
y_train_matrix = np.load('./remote_data/100x100x100_C/'+'y_train_matrix.npy') # shape (n1, n2, n3)
interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
os.makedirs('./data/linear/',exist_ok=True)
with open('./data/linear/new_interpolate3D_100.pickle','wb') as f:
    pickle.dump(interpolate3D,f,pickle.HIGHEST_PROTOCOL)

start = time.time()
interpolate3D(X_test, method='cubic') # Requires 38.1 GiB
print(time.time() - start)

with open('./data/linear/new_interpolate3D_100.pickle','rb') as f:
    interpolate3D = pickle.load(f)

# (10x10x10)
p_test = np.linspace(0.8,0.9999,10)
phi_test = np.linspace(0,1,10) # shape(m1,)
gamma_test = np.linspace(0.5,2,10) # shape(m2,)
X_test = np.vstack(np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')).reshape(3,-1).T # shape(n1*n2*n3, 3)

start = time.time()
cubic_pred = interpolate3D(X_test, method='cubic')
print(time.time() - start) # 0.4302 seconds

start = time.time()
linear_pred = interpolate3D(X_test)
print(time.time() - start) # 0.00987 seconds

start = time.time()
y = qRW_Newton(X_test[:,0], X_test[:,1], X_test[:,2], 100)
print(time.time() - start) # 1.251 seconds

abs_rel_err(cubic_pred, y) # 4.830694877677456e-12
abs_rel_err(linear_pred, y) # 4.83066888350064e-12

# (20x20x20)
p_test = np.linspace(0.8,0.9999,20)
phi_test = np.linspace(0,1,20) # shape(m1,)
gamma_test = np.linspace(0.5,2,20) # shape(m2,)
X_test = np.vstack(np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')).reshape(3,-1).T # shape(n1*n2*n3, 3)

start = time.time()
cubic_pred = interpolate3D(X_test, method='cubic')
print(time.time() - start) # 2.6015350818634033

start = time.time()
linear_pred = interpolate3D(X_test)
print(time.time() - start) # 0.010646581649780273

start = time.time()
y = qRW_Newton(X_test[:,0], X_test[:,1], X_test[:,2], 100)
print(time.time() - start) # 8.269309520721436

abs_rel_err(cubic_pred, y) # 0.016273399991283292
abs_rel_err(linear_pred, y) # 0.0010940535251151588

fig, ax = plt.subplots()
rel_err_cubic = []
rel_err_linear = []
for p in p_test:
    idx = np.where(X_test[:,0] == p)
    rel_err_cubic.append(abs_rel_err(cubic_pred[idx], y[idx]))
    rel_err_linear.append(abs_rel_err(linear_pred[idx], y[idx]))
    # MSEs.append(MSE(y_pred[idx],y_test_matrix_ravel[idx]))
ax.plot(p_test, rel_err_cubic, marker='o',label='cubic')
ax.plot(p_test, rel_err_linear, marker='o', label='linear')

ax.set_ylabel('in %')
ax.set_xlabel('p')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.xlim([0.95,0.99])
# plt.ylim([0,1])
plt.show()
fig.savefig('name.pdf')

# (30x30x30)
p_test = np.linspace(0.8,0.9999,30)
phi_test = np.linspace(0,1,30) # shape(m1,)
gamma_test = np.linspace(0.5,2,30) # shape(m2,)
X_test = np.vstack(np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')).reshape(3,-1).T # shape(n1*n2*n3, 3)

start = time.time()
cubic_pred = interpolate3D(X_test, method='cubic')
print(time.time() - start) # 8.520947456359863

start = time.time()
linear_pred = interpolate3D(X_test)
print(time.time() - start) # 0.004732608795166016

start = time.time()
y = qRW_Newton(X_test[:,0], X_test[:,1], X_test[:,2], 100)
print(time.time() - start) # 27.945157527923584

abs_rel_err(cubic_pred, y) # 0.08094726981626765
abs_rel_err(linear_pred, y) # 0.0017697532587528503

fig, ax = plt.subplots()
rel_err_cubic = []
rel_err_linear = []
for p in p_test:
    idx = np.where(X_test[:,0] == p)
    rel_err_cubic.append(abs_rel_err(cubic_pred[idx], y[idx]))
    rel_err_linear.append(abs_rel_err(linear_pred[idx], y[idx]))
    # MSEs.append(MSE(y_pred[idx],y_test_matrix_ravel[idx]))
ax.plot(p_test, rel_err_cubic, marker='o',label='cubic')
ax.plot(p_test, rel_err_linear, marker='o', label='linear')

ax.set_ylabel('in %')
ax.set_xlabel('p')
legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
# plt.xlim([0.95,0.99])
# plt.ylim([0,1])
plt.show()
fig.savefig('30x30x30 cubic.pdf')