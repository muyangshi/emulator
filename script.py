"""
Plotting the results of RBF_50_C
"""
import math
import time
import os
import pickle

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt

y_pred_all = np.load('./remote_data/RBF_50_C/y_pred_all.npy')
y = np.load('./remote_data/RBF_50_C/y.npy')
X = np.load('./remote_data/RBF_50_C/X.npy')

p = np.linspace(0.8,0.9999,100)
phi = np.linspace(1e-10,1,25) # shape = (v1,)
gamma = np.linspace(0.5,2,25) # shape = (v2,)

p49 = p[49]
idx49 = np.where(X[:,0] == p49)
X49 = X[idx49]
y49 = y[idx49]
y_pred49 = y_pred_all[idx49]


ax = plt.axes(projection='3d')
true_points = ax.scatter(X49[:,1].ravel(), X49[:,2].ravel(), y49.ravel(), color = 'slateblue', marker='^')
pred_surf = ax.plot_trisurf(X49[:,1].ravel(), X49[:,2].ravel(), y_pred49.ravel(), color = 'orange')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel(r'$F^{-1}$')
ax.set_xlim([0,0.8])
ax.set_ylim([0.5,1.4])
ax.set_zlim([0,1000])
plt.title(f'p = {p49}')
# plt.savefig('test.pdf')
plt.show()


################################################################################################################################
# %%
# imports
# -------
import math
import time
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt

# import gp_emulator
import GPy

# %%
# Define Helper Functions
# -----------------------

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
# --------------------------------
y_test_pred_sparse = np.load('./remote_data/sparse/y_test_pred_sparse.npy') # shape (512000, 1)

MSE(y_test_pred_sparse.ravel(), y_test_matrix_ravel)
abs_rel_err(y_test_pred_sparse.ravel(), y_test_matrix_ravel)

# %%
# ----------------------------
p49 = p_test[49]
idx49 = np.where(X_test[:,0] == p49)
X49 = X_test[idx49]
y49 = y_test_matrix_ravel[idx49]
y_pred49 = y_test_pred_sparse.ravel()[idx49]
ax = plt.axes(projection='3d')
true_points = ax.scatter(X49[:,1].ravel(), X49[:,2].ravel(), y49.ravel(), color = 'slateblue', marker='^')
pred_surf = ax.plot_trisurf(X49[:,1].ravel(), X49[:,2].ravel(), y_pred49.ravel(), color = 'orange')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel(r'$F^{-1}$')
# ax.set_xlim([0,0.8])
# ax.set_ylim([0.5,1.4])
# ax.set_zlim([0,400])
plt.title(f'p = {p49}')
# plt.savefig('test.pdf')
plt.show()