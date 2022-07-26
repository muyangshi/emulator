from model_sim import *

import time
import pickle

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern

from scipy.interpolate import RegularGridInterpolator

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

phi_train = np.linspace(0,1,100) # shape(n1,)
gamma_train = np.linspace(0.5,2,100) # shape(n2,)
phi_train_g, gamma_train_g = np.meshgrid(phi_train, gamma_train, indexing='ij')
X_train = np.vstack([phi_train_g,gamma_train_g]).reshape(2,-1).T # shape(n1*n2,2)
X_train_p, X_train_g = np.split(X_train,2,-1) # each shape (n_i,1)
y_train_matrix = qRW_Newton(0.9,phi_train_g, gamma_train_g, 100) # shape(n1, n2)
interpolate2D = RegularGridInterpolator((phi_train, gamma_train), y_train_matrix)

phi_test = np.linspace(0,1,60) # shape(m1,)
gamma_test = np.linspace(0.5,2,60) # shape(m2,)
phi_test_g, gamma_test_g = np.meshgrid(phi_test, gamma_test, indexing='ij')
# Do this, becuase the ordering the points would be consistent
# with the test_pts, which will be input to the RegularGridInterpolator
X_test = np.vstack([phi_test_g,gamma_test_g]).reshape(2,-1).T
X_test_p,X_test_g = np.split(X_test,2,-1)
# y_test = qRW_Newton(0.9,phi_test_g, gamma_test_g, 100)
y_test = qRW_Newton(0.9,X_test_p,X_test_g,100) # shape(m1*m2,1)

train_pts = X_train
test_pts = X_test

# interpolate2D(train_pts) - y_train_matrix.ravel()
interpolate2D(test_pts) - y_test.ravel()
MSE(interpolate2D(test_pts),y_test.ravel())

fig = plt.figure()
ax = plt.axes(projection='3d')
points = ax.scatter(X_test_p.ravel(), X_test_g.ravel(), y_test.ravel(),color='slateblue',marker="^")
# train_points = ax.scatter(X_train_p.ravel(),X_train_g.ravel(),y_train_matrix.ravel(),color='orange',marker='o',alpha=0.2)
ax.plot_surface(phi_train_g, gamma_train_g, y_train_matrix,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
plt.show()

###########################################################################################################################

# p_train = np.linspace(0.8,0.9999,100)
# phi_train = np.linspace(0,1,100) # shape(n1,)
# gamma_train = np.linspace(0.5,2,100) # shape(n2,)
# p_train_g, phi_train_g, gamma_train_g = np.meshgrid(phi_train, gamma_train, indexing='ij')
# X_train = np.vstack([phi_train_g,gamma_train_g]).reshape(2,-1).T # shape(n1*n2,2)
# X_train_p, X_train_g = np.split(X_train,2,-1) # each shape (n_i,1)
# y_train_matrix = qRW_Newton(0.9,phi_train_g, gamma_train_g, 100) # shape(n1, n2)
# interpolate2D = RegularGridInterpolator((phi_train, gamma_train), y_train_matrix)


p_train = np.linspace(0.8,0.9999,100) # shape (n1,)
phi_train = np.linspace(0,1,100) # shape (n2, )
gamma_train = np.linspace(0.5,2,100) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/100x100x100/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

path = './remote_data/80x80x80/'
y_test_matrix = np.load(path+'y_test_matrix.npy')
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )
y_test_vector = np.load(path+'y_test_vector.npy') # shape(n1*n2*n3, 1)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
interpolate3D_near = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix, method = 'nearest')

MSE(interpolate3D(X_test),y_test_matrix_ravel) # 1.133e8
MSE(interpolate3D(X_train),y_train_matrix_ravel) # 0.0

MSE(interpolate3D_near(X_test), y_test_matrix_ravel) # 3.766e10
MSE(interpolate3D(X_test, method = 'nearest'),y_test_matrix_ravel) # same as interpolate3D_near

MSE(interpolate3D_near(X_train), y_train_matrix_ravel)

start = time.time()
interpolate3D(X_train)
time.time() - start

# Graphics

idx = np.where(X_test[:,0] == p_test[79])
mini_idx = idx[0][0::5]

fig = plt.figure()
ax = plt.axes(projection='3d')
points = ax.scatter(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx], y_test_matrix_ravel[mini_idx],color='slateblue',marker="^")
surf = ax.plot_trisurf(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx], interpolate3D(X_test[mini_idx]),color='orange',alpha=0.2)
# train_points = ax.scatter(X_train_p.ravel(),X_train_g.ravel(),y_train_matrix.ravel(),color='orange',marker='o',alpha=0.2)
# ax.plot_surface(phi_train_g, gamma_train_g, y_train_matrix,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel(r'$F^{-1}$')
plt.show()
# Pretty close, in terms of the actually quantile value, but the deviation is big

sum((interpolate3D(X_test[mini_idx]) - y_test_matrix_ravel[mini_idx])**2)/1280
np.argmin(interpolate3D(X_test[mini_idx]) - y_test_matrix_ravel[mini_idx]) # 32
mini_idx[32] # 505760
interpolate3D(X_test[505760])
qRW_Newton(X_test[505760][0],X_test[505760][1],X_test[505760][2],100)

np.argmin(interpolate3D(X_test) - y_test_matrix_ravel) # 506214
interpolate3D(X_test[506214])
qRW_Newton(X_test[506214][0],X_test[506214][1],X_test[506214][2],100)