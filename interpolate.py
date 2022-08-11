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

from scipy.interpolate import RegularGridInterpolator

# %%
# define helper functions
# -----------------------------------------------------------------
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

def abs_rel_err(y_pred, y_true):
    abs_rel_res = abs(y_true - y_pred)/y_true
    return sum(abs_rel_res)/len(abs_rel_res)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
# X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)
path = './remote_data/80x80x80_C/'
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )


# %%
# 2D
# ------------------------------------------------------------------
phi_train = np.linspace(0,1,100) # shape(n1,)
gamma_train = np.linspace(0.5,2,100) # shape(n2,)
phi_train_g, gamma_train_g = np.meshgrid(phi_train, gamma_train, indexing='ij')
X_train = np.vstack([phi_train_g,gamma_train_g]).reshape(2,-1).T # shape(n1*n2,2)
X_train_p, X_train_g = np.split(X_train,2,-1) # each shape (n_i,1)
y_train_matrix = qRW_Newton(0.9,phi_train_g, gamma_train_g, 100) # shape(n1, n2)
interpolate2D = RegularGridInterpolator((phi_train, gamma_train), y_train_matrix)

phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
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

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# points = ax.scatter(X_test_p.ravel(), X_test_g.ravel(), y_test.ravel(),color='slateblue',marker="^")
# # train_points = ax.scatter(X_train_p.ravel(),X_train_g.ravel(),y_train_matrix.ravel(),color='orange',marker='o',alpha=0.2)
# ax.plot_surface(phi_train_g, gamma_train_g, y_train_matrix,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
# plt.show()

###########################################################################################################################

# %%
# 3D Interpolator 100x100x100
# -----------------------------------------------------------------
p_train = np.linspace(0.8,0.9999,100) # shape (n1,)
phi_train = np.linspace(0,1,100) # shape (n2, )
gamma_train = np.linspace(0.5,2,5) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/100x100x100_C/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

path = './remote_data/80x80x80_C/'
y_test_matrix = np.load(path+'y_test_matrix.npy')
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )
y_test_vector = np.load(path+'y_test_vector.npy') # shape(n1*n2*n3, 1)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)

with open('./data/linear/interpolate3D_100.pickle','wb') as f:
    pickle.dump(interpolate3D,f,pickle.HIGHEST_PROTOCOL)

interpolate3D_near = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix, method = 'nearest')
# Specify the method only changes how values are calculated
# Using nearest neighbor for interpolation lead to worse MSE

MSE(interpolate3D(X_test),y_test_matrix_ravel) # 1.133e8
MSE(interpolate3D(X_train),y_train_matrix_ravel) # 0.0

MSE(interpolate3D_near(X_test), y_test_matrix_ravel) # 3.766e10
MSE(interpolate3D(X_test, method = 'nearest'),y_test_matrix_ravel) # same as interpolate3D_near

MSE(interpolate3D_near(X_train), y_train_matrix_ravel)

abs_rel_res = abs(y_test_matrix_ravel - interpolate3D(X_test))/y_test_matrix_ravel
sum(abs_rel_res)/len(abs_rel_res)

start = time.time()
interpolate3D(X_test)
time.time() - start

# %%
# Graphics
# ---------------------------------------------------------------

# looking at a slice of specific p: p_test[79]
idx = np.where(X_test[:,0] == p_test[79])
# take a point once every 5 points
# so plotting can be much less comupation heavy
mini_idx = idx[0][0::5]

fig = plt.figure()
ax = plt.axes(projection='3d')
res79_graph = y_test_matrix_ravel[mini_idx] - interpolate3D(X_test[mini_idx])
# points = ax.scatter(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx], y_test_matrix_ravel[mini_idx],color='slateblue',marker="^")
# surf = ax.plot_trisurf(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx], interpolate3D(X_test[mini_idx]),color='orange',alpha=0.2)
surf = ax.scatter(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx],res79_graph)
# train_points = ax.scatter(X_train_p.ravel(),X_train_g.ravel(),y_train_matrix.ravel(),color='orange',marker='o',alpha=0.2)
# ax.plot_surface(phi_train_g, gamma_train_g, y_train_matrix,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel(r'$F^{-1}$')
plt.show()
# Pretty close, in terms of the actually quantile value, but the deviation is big


# # %%
# # Evaluate
# # ------------------------------------------------------------

# sum((interpolate3D(X_test[mini_idx]) - y_test_matrix_ravel[mini_idx])**2)/1280
# np.argmin(interpolate3D(X_test[mini_idx]) - y_test_matrix_ravel[mini_idx]) # 32
# mini_idx[32] # 505760
# interpolate3D(X_test[505760])
# qRW_Newton(X_test[505760][0],X_test[505760][1],X_test[505760][2],100)

# np.argmin(interpolate3D(X_test) - y_test_matrix_ravel) # 506214
# interpolate3D(X_test[506214])
# qRW_Newton(X_test[506214][0],X_test[506214][1],X_test[506214][2],100)

# np.argmax(interpolate3D(X_test[mini_idx]) - y_test_matrix_ravel[mini_idx]) # 1247
# mini_idx[1247] # 511835
# interpolate3D(X_test[511835]) # 1.94876501e+08
# qRW_Newton(X_test[511835][0],X_test[511835][1],X_test[511835][2],100) # 1.9411452e+08
# interpolate3D(X_test[511835]) - qRW_Newton(X_test[511835][0],X_test[511835][1],X_test[511835][2],100) # 761980


# %%
# Evaluation
# -----------------------------------------------------------
y_test_pred = interpolate3D(X_test) # interpolated values
res = y_test_matrix_ravel - y_test_pred # residuals
p_test # probability values in the testing dataset


idx0 = np.where(X_test[:,0] == p_test[0])
sum(res[idx0]**2)/len(idx0[0])
# datapoints corresponding to p = p_test[79]
# looking at a slice of the 3D input
idx = np.where(X_test[:,0] == p_test[79])
sum(res[idx]**2)/len(idx[0])
# MSE(y_test_pred[idx],y_test_matrix_ravel[idx]) # same
res79 = res[idx] # residuals corresponding to the slice
qRW_Newton(p_test[79],1,2,100) # scale of the output

np.argmax(res79) # 614
res79[614]
idx[0][614] # 506214
X_test[506214,:] # input that leads to the largest positive residual at p79

np.argmax(res) # also 506214

np.argmin(res79) # 6239
res79[6239]
idx[0][6239] # 511839
X_test[511839,:]

#Note that it's also the largest absolute res
np.argmax(np.abs(res)) # 511839

# %%
# Issue about C++ implemented quantile function
# ----------------------------------------------
"""
Looking at res79 index80-150, we see that the residuals are all the same
I wonder what's happening
"""
res79[80] # 42.447213538571134
res79[81] # 42.447213538571134
res79[159]# 42.447213538571134
idx[0][80] # 505680
idx[0][81] # 505681
idx[0][159] # 505759
X_test[505680,:]
X_test[505681,:]
X_test[505759,:]

# They are on the same non-sloped line so 
# interpolated quantile are the same
# p = 0.9999
# phi = X_test[505680,1] = 0.012658227848101266
# gamma is different for the three points
interpolate3D(X_test[505680,:])
interpolate3D(X_test[505681,:])
interpolate3D(X_test[505759,:])

# weird, still all same
qRW_Newton(0.9999,X_test[505680,1], X_test[(505680,505681,505759),2],100)

# python implementation now seems ok
qRW_Newton_py(0.9999,X_test[505680,1], X_test[(505680,505681,505759),2],100)


# %%
# More Grid Points 200x200x200
# --------------------------------------------------------------------
"""
Using 200x200x200 grid points to construct the linear interpolator
On the same testing dataset of 80x80x80, MSE decreases from 1e8 to 7e6
"""
p_train = np.linspace(0.8,0.9999,200) # shape (n1,)
phi_train = np.linspace(0,1,200) # shape (n2, )
gamma_train = np.linspace(0.5,2,200) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/m200x200x200_C/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
# y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

path = './remote_data/80x80x80_C/'
y_test_matrix = np.load(path+'y_test_matrix.npy')
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )
y_test_vector = np.load(path+'y_test_vector.npy') # shape(n1*n2*n3, 1)


interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)

with open('./data/linear/interpolate3D_200.pickle','wb') as f:
    pickle.dump(interpolate3D,f,pickle.HIGHEST_PROTOCOL)

start = time.time()
interpolate3D(X_test)
print('200x200x200 training', time.time() - start)

MSE(interpolate3D(X_test),y_test_matrix_ravel) # 7.522e6
MSE(interpolate3D(X_train),y_train_matrix_ravel) # 0.0

res = abs(y_test_matrix_ravel - interpolate3D(X_test))
rel_res = res/y_test_matrix_ravel
sum(rel_res)/len(rel_res)

abs_rel_err(interpolate3D(X_test),y_test_matrix_ravel) # 0.001052293895840019

np.max(rel_res)
np.argmax(rel_res)
X_test[506125,:]

interpolate3D(X_test[506125,:])
qRW_Newton(X_test[506125,0],X_test[506125,1],X_test[506125,2],100)
qRW_Newton_py(X_test[506125,0],X_test[506125,1],X_test[506125,2],100)

inter_idx = np.where(X_test[:,0] != 0.9999)
X_test_inter = X_test[inter_idx]
res_inter = abs(y_test_matrix_ravel[inter_idx] - interpolate3D(X_test[inter_idx,:]))
# rel_res_inter = 

# %%
# 400x400x400
# --------------------------------------------------------------------------------------
p_train = np.linspace(0.8,0.9999,400) # shape (n1,)
phi_train = np.linspace(0,1,400) # shape (n2, )
gamma_train = np.linspace(0.5,2,400) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/400x400x400/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
# y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

path = './remote_data/80x80x80_C/'
y_test_matrix = np.load(path+'y_test_matrix.npy')
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )
y_test_vector = np.load(path+'y_test_vector.npy') # shape(n1*n2*n3, 1)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)

with open('./data/linear/interpolate3D_400.pickle','wb') as f:
    pickle.dump(interpolate3D,f,pickle.HIGHEST_PROTOCOL)

MSE(interpolate3D(X_test),y_test_matrix_ravel)
abs_rel_err(interpolate3D(X_test),y_test_matrix_ravel)

start = time.time()
interpolate3D(X_test)
print(time.time() - start)

# %%
# 500x500x500
# ---------------------------------------------------------------------------------------
p_train = np.linspace(0.8,0.9999,500) # shape (n1,)
phi_train = np.linspace(0,1,500) # shape (n2, )
gamma_train = np.linspace(0.5,2,500) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/m500x500x500_C/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
# y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

path = './remote_data/80x80x80_C/'
y_test_matrix = np.load(path+'y_test_matrix.npy')
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )
y_test_vector = np.load(path+'y_test_vector.npy') # shape(n1*n2*n3, 1)


interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
os.makedirs('./data/linear/',exist_ok=True)
with open('./data/linear/interpolate3D_500.pickle','wb') as f:
    pickle.dump(interpolate3D,f,pickle.HIGHEST_PROTOCOL)

MSE(interpolate3D(X_test),y_test_matrix_ravel)

absres = abs(y_test_matrix_ravel - interpolate3D(X_test))/y_test_matrix_ravel
sum(absres)/len(absres)

y200 = np.load('./remote_data/200x200x200_C/y_train_matrix.npy')
y200m = np.load('./remote_data/m200x200x200_C/y_train_matrix.npy')


# %%
# Uneven 100 + 300
# -----------------------------------------------------------------------------------------------
p_train = np.append(np.linspace(0.8,0.9,100),np.linspace(0.9,0.99999,301)) # shape (n1,)
p_train = np.delete(p_train, 100) # remove the duplicate
phi_train = np.append(np.linspace(1e-10,0.6,100),np.linspace(0.6,1,301)) # shape (n2, )
phi_train = np.delete(phi_train,100) # remove the duplicate
gamma_train = np.append(np.linspace(0.5,1.5,100), np.linspace(1.5,2,301)) # shape (n3, )
gamma_train = np.delete(gamma_train,100) # remove the duplicate
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
# X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/100_300/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
# y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
# y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
with open('./data/linear/interpolate3D_100_300.pickle','wb') as f:
    pickle.dump(interpolate3D, f, pickle.HIGHEST_PROTOCOL)

MSE(interpolate3D(X_train), y_train_matrix)

# %%
# Uneven 100 + 400
# -----------------------------------------------------------------------------------------------
p_train = np.append(np.linspace(0.8,0.9,100),np.linspace(0.9,0.99999,401)) # shape (n1,)
p_train = np.delete(p_train, 100) # remove the duplicate
phi_train = np.append(np.linspace(1e-10,0.6,100),np.linspace(0.6,1,401)) # shape (n2, )
phi_train = np.delete(phi_train,100) # remove the duplicate
gamma_train = np.append(np.linspace(0.5,1.5,100), np.linspace(1.5,2,401)) # shape (n3, )
gamma_train = np.delete(gamma_train,100) # remove the duplicate
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)

path = './remote_data/100_400/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
# y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
# y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
with open('./data/linear/interpolate3D_100_400.pickle','wb') as f:
    pickle.dump(interpolate3D, f, pickle.HIGHEST_PROTOCOL)

# np.save(path+'y_train_vector',y_train_vector) # y_train_vector.ravel() = y_train_matrix_ravel, using indexing='ij'

# %%
# Uneven 100 + 400 version 2
# -----------------------------------------------------------------------------------------------
p_train = np.append(np.linspace(0.8,0.9,100),np.linspace(0.9,0.9999,401)) # shape (n1,)
p_train = np.delete(p_train, 100) # remove the duplicate
phi_train = np.append(np.linspace(0,0.6,100),np.linspace(0.6,1,401)) # shape (n2, )
phi_train = np.delete(phi_train,100) # remove the duplicate
gamma_train = np.append(np.linspace(0.5,1.5,100), np.linspace(1.5,2,401)) # shape (n3, )
gamma_train = np.delete(gamma_train,100) # remove the duplicate
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

path = './remote_data/100_400_2/'
y_train_matrix = np.load(path+'y_train_matrix.npy') # shape (n1, n2, n3)
# y_train_matrix_ravel = np.load(path+'y_train_matrix_ravel.npy') # shape (n1*n2*n3, )
# y_train_vector = np.load(path+'y_train_vector.npy') # shape(n1*n2*n3, 1)

interpolate3D = RegularGridInterpolator((p_train, phi_train, gamma_train), y_train_matrix)
with open('./data/linear/interpolate3D_100_400_2.pickle','wb') as f:
    pickle.dump(interpolate3D, f, pickle.HIGHEST_PROTOCOL)

# np.save(path+'y_train_vector',y_train_vector) # y_train_vector.ravel() = y_train_matrix_ravel, using indexing='ij'


# %%
# Time Evaluation
# -----------------------------------------------------------------------------------------------
with open('./data/linear/interpolate3D_100_400_2.pickle','rb') as f:
    interpolate3D = pickle.load(f)
n_points = [10,50,100,200,300]
for n in n_points:
    print(n)
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
p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
path = './remote_data/80x80x80_C/'
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )

fig, ax = plt.subplots()

models = ['interpolate3D_100.pickle',
            'interpolate3D_200.pickle',
            'interpolate3D_400.pickle',
            'interpolate3D_500.pickle',
            'interpolate3D_100_300_2.pickle',
            'interpolate3D_100_400_2.pickle']
models = ['interpolate3D_500.pickle',
            'interpolate3D_100_300_2.pickle',
            'interpolate3D_100_400_2.pickle']
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

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')
plt.show()





# %%
# Error Evaluation
# ----------------------------------------------------------------------------------------------
with open('./data/linear/interpolate3D_100_300.pickle','rb') as f:
    interpolate3D = pickle.load(f)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)

idx = np.where(X_test[:,1] != 0)

path = './remote_data/80x80x80_C/'
# y_test_matrix = np.load(path+'y_test_matrix.npy')
y_test_matrix_ravel = np.load(path+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )

# interpolate3D(X_test[idx]).shape

MSE(interpolate3D(X_test[idx]),y_test_matrix_ravel[idx])
abs_rel_err(interpolate3D(X_test[idx]),y_test_matrix_ravel[idx])

X_test = X_test[idx]
y_test_matrix_ravel = y_test_matrix_ravel[idx]

# looking at a slice of specific p: p_test[79]
idx = np.where(X_test[:,0] == p_test[79])
# take a point once every 5 points
# so plotting can be much less comupation heavy
mini_idx = idx[0][0::10]

fig = plt.figure()
ax = plt.axes(projection='3d')
# res79_graph = y_test_matrix_ravel[mini_idx] - interpolate3D(X_test[mini_idx])
# res_points = ax.scatter(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx],res79_graph)
points = ax.scatter(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx], y_test_matrix_ravel[mini_idx],color='slateblue',marker="^")
surf = ax.plot_trisurf(X_test_phi.ravel()[mini_idx], X_test_gamma.ravel()[mini_idx], interpolate3D(X_test[mini_idx]),color='orange',alpha=0.2)
# train_points = ax.scatter(X_train_p.ravel(),X_train_g.ravel(),y_train_matrix.ravel(),color='orange',marker='o',alpha=0.2)
# ax.plot_surface(phi_train_g, gamma_train_g, y_train_matrix,rstride=1,cstride=1,cmap='viridis',edgecolor='none')
ax.set_xlabel(r"$\phi$")
ax.set_ylabel(r'$\gamma$')
ax.set_zlabel(r'$F^{-1}$')
plt.show()



p = X_test[511919,0]
phi = X_test[511919,1]
gamma = X_test[511919,2]
start = time.time()
qRW_Newton(p,phi,gamma,100)
print(time.time() - start) # 0.012285947799682617 seconds

start = time.time()
interpolate3D([p,phi,gamma])
print(time.time() - start) # 0.004333972930908203 seconds

start = time.time()
interpolate3D([p,phi,gamma], method='cubic')
print(time.time() - start) # 5.547595262527466 seconds