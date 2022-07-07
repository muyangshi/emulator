# n_tries
# n_restarts_optimizer
# training_size

# %%
# imports
# -------
from model_sim import *

import math
import time
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

import gp_emulator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern
from sklearn.model_selection import KFold

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

def save_scikit_emulator(name: str,obj):
    np.savez(name,obj)

def load_scikit_emulator(file):
    loaded_npzfile = np.load(file, allow_pickle=True)
    loaded_gaussian_process = loaded_npzfile['arr_0'][()]
    return loaded_gaussian_process

def SSE(y_pred, y_true):
    return sum((y_pred - y_true)**2)

def MSE(y_pred, y_true):
    return SSE(y_pred, y_true)/y_true.size

def train_gp(X_train, y_train):
    start_time = time.time()
    gp = gp_emulator.GaussianProcess(inputs = X_train, targets = y_train)
    gp.learn_hyperparameters(n_tries = 5, verbose = False)
    print('gp_emulator done', time.time() - start_time)
    return gp

def train_scikit(X_train, y_train, kernel):
    start_time = time.time()
    gp_scikit = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 5)
    gp_scikit.fit(X_train, y_train)
    print('scikit done', time.time() - start_time)
    return gp_scikit

def pack_MSE(type, gaussian_process, X, X_test, X_train, y, y_test, y_train):
    """
    return [overall, test, train] MSE values
    """
    if type == 'gp_emulator':
        y_pred_all = gaussian_process.predict(X, do_unc=False, do_deriv=False)[0]
        y_pred_test = gaussian_process.predict(X_test, do_unc=False, do_deriv=False)[0]
        y_pred_train = gaussian_process.predict(X_train, do_unc=False, do_deriv=False)[0]
    elif type == 'scikit':
        y_pred_all = gaussian_process.predict(X, return_std = False)
        y_pred_test = gaussian_process.predict(X_test, return_std = False)
        y_pred_train = gaussian_process.predict(X_train, return_std = False)
    else:
        raise Exception('gp_emulator or scikit')
    return [MSE(y_pred_all,y), MSE(y_pred_test,y_test), MSE(y_pred_train,y_train)]

# %%
###########
# DATASET #
###########

# %%
# Dataset (X) Generation
# ------------------
# Specify parameter range
phis = np.linspace(0,1,100) # shape = (v1,)
gammas = np.linspace(1,2,100) # shape = (v2,)
# create grid of parameter 
x_grid = NDimCoord(phis,gammas) # shape = (v1*v2, 2)
phi_coor, gamma_coor = np.split(x_grid,x_grid.shape[1],1) # each shape = (v1*v2, 1)

# %%
# Calculate True y
# ----------------
y = qRW_Newton(0.9,phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)

# %%
# Save Generated Data
# -------------------
data = np.hstack((x_grid,y))
np.save('./data/data',data)
np.save('./data/x_grid',x_grid)
np.save('./data/y',y)
np.save('./data/phi_coor',phi_coor)
np.save('./data/gamma_coor',gamma_coor)

# %%
# Load Generated Data
# -------------------
data = np.load('./data/data.npy')
y = np.load('./data/y.npy')
x_grid = np.load('./data/x_grid.npy')
phi_coor = np.load('./data/phi_coor.npy')
gamma_coor = np.load('./data/gamma_coor.npy')
X = x_grid

# %%
# Splitting Dataset
# -----------------
np.random.seed(42)
n_samples = y.size
training_size = math.floor(n_samples * 0.5)
training_indices = np.random.choice(n_samples,training_size,replace = False)
testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
X_train, y_train = x_grid[training_indices], y[training_indices]
X_test, y_test = x_grid[testing_indices], y[testing_indices]

# %%
# Save Splitted Data
# ------------------
np.save('./data/X_train',X_train)
np.save('./data/y_train',y_train)
np.save('./data/X_test',X_test)
np.save('./data/y_test',y_test)

# %%
# Load Splitted Data
# ------------------
X_train = np.load('./data/X_train.npy')
y_train = np.load('./data/y_train.npy')
X_test = np.load('./data/X_test.npy')
y_test = np.load('./data/y_test.npy')

# %%
############
# TRAINING #
############

# %%
# gp_emulator training
# --------------------
gp = gp_emulator.GaussianProcess(inputs=X_train,targets=y_train)
gp.learn_hyperparameters(n_tries = 10, verbose = False)
gp.save_emulator('./data/gp') # saved as npz file
# save predictions
gp_y_pred_all, gp_y_unc_all, _ = gp.predict(x_grid, do_unc = True, do_deriv = False)
np.save('./data/gp_y_pred_all',gp_y_pred_all)
gp_y_pred_train, gp_y_unc_train, _ = gp.predict(X_train, do_unc = True, do_deriv = False)
np.save('./data/gp_y_pred_train',gp_y_pred_train)
gp_y_pred_test, gp_y_unc_test, _ = gp.predict(X_test, do_unc = True, do_deriv= False)
np.save('./data/gp_y_pred_test',gp_y_pred_test)

# %%
# Scikit RBF Training
# -------------------
kernel_RBF = RBF()
gp_scikit_RBF = GaussianProcessRegressor(kernel=kernel_RBF,n_restarts_optimizer=10)
gp_scikit_RBF.fit(X_train,y_train)
# gp_scikit_RBF.kernel_
gp_scikit_RBF_y_pred_all, gp_scikit_RBF_std_all = gp_scikit_RBF.predict(x_grid, return_std = True)
gp_scikit_RBF_y_pred_train, gp_scikit_RBF_std_train = gp_scikit_RBF.predict(X_train, return_std = True)
gp_scikit_RBF_y_pred_test, gp_scikit_RBF_std_test = gp_scikit_RBF.predict(X_test, return_std = True)
save_scikit_emulator('./data/gp_scikit_RBF',gp_scikit_RBF)
np.save('./data/gp_scikit_RBF_y_pred_all',gp_scikit_RBF_y_pred_all)
np.save('./data/gp_scikit_RBF_y_pred_train',gp_scikit_RBF_y_pred_train)
np.save('./data/gp_scikit_RBF_y_pred_test',gp_scikit_RBF_y_pred_test)

# %%
# Scikit Rational Quadratic Training
# ----------------------------------
kernel_RQ = RationalQuadratic()
gp_scikit_RQ = GaussianProcessRegressor(kernel=kernel_RQ,n_restarts_optimizer=10)
gp_scikit_RQ.fit(X_train,y_train)
# gp_scikit_RQ.kernel_
gp_scikit_RQ_y_pred_all, gp_scikit_RQ_std_all = gp_scikit_RQ.predict(x_grid, return_std = True)
gp_scikit_RQ_y_pred_train, gp_scikit_RQ_std_train = gp_scikit_RQ.predict(X_train, return_std = True)
gp_scikit_RQ_y_pred_test, gp_scikit_RQ_std_test = gp_scikit_RQ.predict(X_test, return_std = True)
save_scikit_emulator('./data/gp_scikit_RQ',gp_scikit_RQ)
np.save('./data/gp_scikit_RQ_y_pred_all',gp_scikit_RQ_y_pred_all)
np.save('./data/gp_scikit_RQ_y_pred_train',gp_scikit_RQ_y_pred_train)
np.save('./data/gp_scikit_RQ_y_pred_test',gp_scikit_RQ_y_pred_test)

# %%
# Scikit Dot Product Training
# ---------------------------
kernel_DP = DotProduct()
gp_scikit_DP = GaussianProcessRegressor(kernel = kernel_DP,n_restarts_optimizer=10)
gp_scikit_DP.fit(X_train,y_train)
# gp_scikit_DP.kernel_
gp_scikit_DP_y_pred_all, gp_scikit_DP_std_all = gp_scikit_DP.predict(x_grid, return_std = True)
gp_scikit_DP_y_pred_train, gp_scikit_DP_std_train = gp_scikit_DP.predict(X_train, return_std = True)
gp_scikit_DP_y_pred_test, gp_scikit_DP_std_test = gp_scikit_DP.predict(X_test, return_std = True)
save_scikit_emulator('./data/gp_scikit_DP',gp_scikit_DP)
np.save('./data/gp_scikit_DP_y_pred_all',gp_scikit_DP_y_pred_all)
np.save('./data/gp_scikit_DP_y_pred_train',gp_scikit_DP_y_pred_train)
np.save('./data/gp_scikit_DP_y_pred_test',gp_scikit_DP_y_pred_test)


# %%
# Scikit Matern v=1/2 Training
# ----------------------------
kernel_M1 = Matern(nu=0.5)
gp_scikit_M1 = GaussianProcessRegressor(kernel = kernel_M1,n_restarts_optimizer=10)
gp_scikit_M1.fit(X_train,y_train)
# gp_scikit_M1.kernel_
gp_scikit_M1_y_pred_all, gp_scikit_M1_std_all = gp_scikit_M1.predict(x_grid, return_std = True)
gp_scikit_M1_y_pred_train, gp_scikit_M1_std_train = gp_scikit_M1.predict(X_train, return_std = True)
gp_scikit_M1_y_pred_test, gp_scikit_M1_std_test = gp_scikit_M1.predict(X_test, return_std = True)
save_scikit_emulator('./data/gp_scikit_M1',gp_scikit_M1)
np.save('./data/gp_scikit_M1_y_pred_all',gp_scikit_M1_y_pred_all)
np.save('./data/gp_scikit_M1_y_pred_train',gp_scikit_M1_y_pred_train)
np.save('./data/gp_scikit_M1_y_pred_test',gp_scikit_M1_y_pred_test)

# %%
# Scikit Matern v = 3/2 Training
# ------------------------------
kernel_M2 = Matern(nu=1.5)
gp_scikit_M2 = GaussianProcessRegressor(kernel = kernel_M2,n_restarts_optimizer=10)
gp_scikit_M2.fit(X_train,y_train)
# gp_scikit_M2.kernel_
gp_scikit_M2_y_pred_all, gp_scikit_M2_std_all = gp_scikit_M2.predict(x_grid, return_std = True)
gp_scikit_M2_y_pred_train, gp_scikit_M2_std_train = gp_scikit_M2.predict(X_train, return_std = True)
gp_scikit_M2_y_pred_test, gp_scikit_M2_std_test = gp_scikit_M2.predict(X_test, return_std = True)
save_scikit_emulator('./data/gp_scikit_M2',gp_scikit_M2)
np.save('./data/gp_scikit_M2_y_pred_all',gp_scikit_M2_y_pred_all)
np.save('./data/gp_scikit_M2_y_pred_train',gp_scikit_M2_y_pred_train)
np.save('./data/gp_scikit_M2_y_pred_test',gp_scikit_M2_y_pred_test)

# %%
# Scikit Matern v = 5/2 Training
# ------------------------------
kernel_M3 = Matern(nu=2.5)
gp_scikit_M3 = GaussianProcessRegressor(kernel = kernel_M3,n_restarts_optimizer=10)
gp_scikit_M3.fit(X_train,y_train)
# gp_scikit_M3.kernel_
gp_scikit_M3_y_pred_all, gp_scikit_M3_std_all = gp_scikit_M3.predict(x_grid, return_std = True)
gp_scikit_M3_y_pred_train, gp_scikit_M3_std_train = gp_scikit_M3.predict(X_train, return_std = True)
gp_scikit_M3_y_pred_test, gp_scikit_M3_std_test = gp_scikit_M3.predict(X_test, return_std = True)
save_scikit_emulator('./data/gp_scikit_M3',gp_scikit_M3)
np.save('./data/gp_scikit_M3_y_pred_all',gp_scikit_M3_y_pred_all)
np.save('./data/gp_scikit_M3_y_pred_train',gp_scikit_M3_y_pred_train)
np.save('./data/gp_scikit_M3_y_pred_test',gp_scikit_M3_y_pred_test)

# %%
##############
# EVALUATION #
##############

# %%
# Load gp_emulator
# ----------------
gp = gp_emulator.GaussianProcess(emulator_file='./data/gp.npz')
gp_y_pred_all = np.load('./data/gp_y_pred_all.npy')
gp_y_pred_train = np.load('./data/gp_y_pred_train.npy')
gp_y_pred_test = np.load('./data/gp_y_pred_test.npy')

# %%
# Load Scikit RBF
# ----------------------------
gp_scikit_RBF = load_scikit_emulator('./data/gp_scikit_RBF.npz')
gp_scikit_RBF_y_pred_all = np.load('./data/gp_scikit_RBF_y_pred_all.npy')
gp_scikit_RBF_y_pred_train = np.load('./data/gp_scikit_RBF_y_pred_train.npy')
gp_scikit_RBF_y_pred_test = np.load('./data/gp_scikit_RBF_y_pred_test.npy')
# np.savez('gaussian_process',gaussian_process)
# loaded_npzfile = np.load('gaussian_process.npz',allow_pickle=True)
# loaded_gaussian_process = loaded_npzfile['arr_0'][()]
# test = loaded_gaussian_process.predict(x_grid, return_std=False) # same!

# %%
# Load Scikit RQ
# --------------
gp_scikit_RQ = load_scikit_emulator('./data/gp_scikit_RQ.npz')
gp_scikit_RQ_y_pred_all = np.load('./data/gp_scikit_RQ_y_pred_all.npy')
gp_scikit_RQ_y_pred_train = np.load('./data/gp_scikit_RQ_y_pred_train.npy')
gp_scikit_RQ_y_pred_test = np.load('./data/gp_scikit_RQ_y_pred_test.npy')

# %%
# Load Scikit Dot Product
# -----------------------
gp_scikit_DP = load_scikit_emulator('./data/gp_scikit_DP.npz')
gp_scikit_DP_y_pred_all = np.load('./data/gp_scikit_DP_y_pred_all.npy')
gp_scikit_DP_y_pred_train = np.load('./data/gp_scikit_DP_y_pred_train.npy')
gp_scikit_DP_y_pred_test = np.load('./data/gp_scikit_DP_y_pred_test.npy')

# %%
# Load Scikit Matern 1/2
# ----------------------
gp_scikit_M1 = load_scikit_emulator('./data/gp_scikit_M1.npz')
gp_scikit_M1_y_pred_all = np.load('./data/gp_scikit_M1_y_pred_all.npy')
gp_scikit_M1_y_pred_train = np.load('./data/gp_scikit_M1_y_pred_train.npy')
gp_scikit_M1_y_pred_test = np.load('./data/gp_scikit_M1_y_pred_test.npy')

# %%
# Load Scikit Matern 3/2
# ----------------------
gp_scikit_M2 = load_scikit_emulator('./data/gp_scikit_M2.npz')
gp_scikit_M2_y_pred_all = np.load('./data/gp_scikit_M2_y_pred_all.npy')
gp_scikit_M2_y_pred_train = np.load('./data/gp_scikit_M2_y_pred_train.npy')
gp_scikit_M2_y_pred_test = np.load('./data/gp_scikit_M2_y_pred_test.npy')

# %%
# Load Scikit Matern 5/2
# ----------------------
gp_scikit_M3 = load_scikit_emulator('./data/gp_scikit_M3.npz')
gp_scikit_M3_y_pred_all = np.load('./data/gp_scikit_M3_y_pred_all.npy')
gp_scikit_M3_y_pred_train = np.load('./data/gp_scikit_M3_y_pred_train.npy')
gp_scikit_M3_y_pred_test = np.load('./data/gp_scikit_M3_y_pred_test.npy')



# %%
# Calculate MSE
# -------------

# gp_emulator
print('gp_emulator')
print('overall',MSE(gp_y_pred_all,y))
print('test',MSE(gp_y_pred_test,y_test))
print('train',MSE(gp_y_pred_train,y_train))

# Scikit RBF
print('Scikit RBF')
print('overall',MSE(gp_scikit_RBF_y_pred_all,y))
print('test',MSE(gp_scikit_RBF_y_pred_test,y_test))
print('train',MSE(gp_scikit_RBF_y_pred_train,y_train))

# Scikit RQ
print('Scikit Rational Quadratic')
print('overall',MSE(gp_scikit_RQ_y_pred_all,y))
print('test',MSE(gp_scikit_RQ_y_pred_test,y_test))
print('train',MSE(gp_scikit_RQ_y_pred_train,y_train))

# Scikit Dot Product
print('Scikit Dot Product')
print('overall',MSE(gp_scikit_DP_y_pred_all,y))
print('test',MSE(gp_scikit_DP_y_pred_test,y_test))
print('train',MSE(gp_scikit_DP_y_pred_train,y_train))

# Scikit Matern 1/2
print('Scikit Matern nu = 0.5')
print('overall',MSE(gp_scikit_M1_y_pred_all,y))
print('test',MSE(gp_scikit_M1_y_pred_test,y_test))
print('train',MSE(gp_scikit_M1_y_pred_train,y_train))

# Scikit Matern 3/2
print('Scikit Matern nu = 1.5')
print('overall',MSE(gp_scikit_M2_y_pred_all,y))
print('test',MSE(gp_scikit_M2_y_pred_test,y_test))
print('train',MSE(gp_scikit_M2_y_pred_train,y_train))

# Scikit Matern 5/2
print('Scikit Matern nu = 2.5')
print('overall',MSE(gp_scikit_M3_y_pred_all,y))
print('test',MSE(gp_scikit_M3_y_pred_test,y_test))
print('train',MSE(gp_scikit_M3_y_pred_train,y_train))


# %%
# Drawing 3D Output Surface
# -------------------------
# gp_emulator
title = '0.9 quantile'
X = phi_coor.ravel()
Y = gamma_coor.ravel()
Z_true = y
Z_pred = gp_y_pred_all
draw3D(title, X, Y, Z_pred, Z_true)


# draw3D('quantile = 0.9',phi_coor.ravel(),gamma_coor.ravel(),test,y)


# X = phi_coor.ravel()
# Y = gamma_coor.ravel()
# Z = mean_prediction
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot_trisurf(X,Y,Z,color='orange') # Predicted
# ax.plot_trisurf(X,Y,y.ravel(),color='blue') # True
# ax.set_title('surface')
# ax.set_xlabel('phi')
# ax.set_ylabel('gamma')
# ax.set_zlabel('inverse')
# plt.show()

# X = phi_coor.ravel()
# Y = gamma_coor.ravel()
# Z = mean_prediction2
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot_trisurf(X,Y,Z,color='orange') # Predicted
# ax.plot_trisurf(X,Y,y.ravel(),color='blue') # True
# ax.set_title('surface')
# ax.set_xlabel('phi')
# ax.set_ylabel('gamma')
# ax.set_zlabel('inverse')
# plt.show()

#########################
## Training Simplified ##
#########################
# %%
# training with 50% of data
# (re-write of training, simplified)
# ----------------------------------

# Training
gp = train_gp(X_train, y_train)
gp_scikit_RBF = train_scikit(X_train, y_train, RBF())
gp_scikit_RQ = train_scikit(X_train, y_train, RationalQuadratic())
# gp_scikit_DP = train_scikit(X_train, y_train, DotProduct())
gp_scikit_M1 = train_scikit(X_train, y_train, Matern(nu=0.5))
gp_scikit_M2 = train_scikit(X_train ,y_train, Matern(nu=1.5))
gp_scikit_M3 = train_scikit(X_train, y_train, Matern(nu=2.5))

# Results
gp_MSE = pack_MSE('gp_emulator', gp, X, X_test, X_train, y, y_test, y_train)
RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
# DP_MSE = pack_MSE('scikit',gp_scikit_DP, X, X_test, X_train, y, y_test, y_train)
M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

gp.save_emulator('./data/training_50_percent/gp')
save_scikit_emulator('./data/training_50_percent/gp_scikit_RBF',gp_scikit_RBF)
save_scikit_emulator('./data/training_50_percent/gp_scikit_RQ',gp_scikit_RQ)
# save_scikit_emulator('./data/training_50_percent/gp_scikit_DP',gp_scikit_DP)
save_scikit_emulator('./data/training_50_percent/gp_scikit_M1',gp_scikit_M1)
save_scikit_emulator('./data/training_50_percent/gp_scikit_M2',gp_scikit_M2)
save_scikit_emulator('./data/training_50_percent/gp_scikit_M3',gp_scikit_M3)

np.save('./data/training_50_percent/gp_MSE',gp_MSE)
np.save('./data/training_50_percent/RBF_MSE',RBF_MSE)
np.save('./data/training_50_percent/RQ_MSE',RQ_MSE)
# np.save('./data/training_50_percent/DP_MSE',DP_MSE)
np.save('./data/training_50_percent/M1_MSE',M1_MSE)
np.save('./data/training_50_percent/M2_MSE',M2_MSE)
np.save('./data/training_50_percent/M3_MSE',M3_MSE)

# Example Graph
gp_scikit_RBF = load_scikit_emulator('./data/training_50_percent/gp_scikit_RBF.npz')
Z_pred = gp_scikit_RBF.predict(X)
title = '0.9 quantile'
X = phi_coor.ravel()
Y = gamma_coor.ravel()
Z_true = y
draw3D(title, X, Y, Z_pred, Z_true)
######################
## Cross Validation ##
######################
# %%

# Place to Store Results:
gp_MSEs, RBF_MSEs, RQ_MSEs, DP_MSEs, M1_MSEs, M2_MSEs, M3_MSEs = [[] for i in range(7)]

kf = KFold(n_splits = 2, shuffle = True, random_state = 42)

for train_index, test_index in kf.split(x_grid):
    # Partitioning the Dataset:
    X_train, X_test = x_grid[train_index,:], x_grid[test_index,:]
    y_train, y_test = y[train_index], y[test_index]

    # Training:
    gp = train_gp(X_train, y_train)
    gp_scikit_RBF = train_scikit(X_train, y_train, RBF())
    gp_scikit_RQ = train_scikit(X_train, y_train, RationalQuadratic())
    gp_scikit_DP = train_scikit(X_train, y_train, DotProduct())
    gp_scikit_M1 = train_scikit(X_train, y_train, Matern(nu=0.5))
    gp_scikit_M2 = train_scikit(X_train ,y_train, Matern(nu=1.5))
    gp_scikit_M3 = train_scikit(X_train, y_train, Matern(nu=2.5))

    # Store Results
    gp_MSEs.append(pack_MSE('gp_emulator', gp, X, X_test, X_train, y, y_test, y_train))
    RBF_MSEs.append(pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train))
    RQ_MSEs.append(pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train))
    DP_MSEs.append(pack_MSE('scikit',gp_scikit_DP, X, X_test, X_train, y, y_test, y_train))
    M1_MSEs.append(pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train))
    M2_MSEs.append(pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train))
    M3_MSEs.append(pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train))

gp_MSE = np.mean(gp_MSEs, axis = 0)
RBF_MSE = np.mean(RBF_MSEs, axis = 0)
RQ_MSE = np.mean(RQ_MSEs, axis = 0)
DP_MSE = np.mean(DP_MSEs, axis = 0)
M1_MSE = np.mean(M1_MSEs, axis = 0)
M2_MSE = np.mean(M2_MSEs, axis = 0)
M3_MSE = np.mean(M3_MSEs, axis = 0)

np.save('./data/CV/gp_MSE',gp_MSE)
np.save('./data/CV/RBF_MSE',RBF_MSE)
np.save('./data/CV/RQ_MSE',RQ_MSE)
np.save('./data/CV/DP_MSE',DP_MSE)
np.save('./data/CV/M1_MSE',M1_MSE)
np.save('./data/CV/M2_MSE',M2_MSE)
np.save('./data/CV/M3_MSE',M3_MSE)


