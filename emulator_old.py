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
# Configure Training Settings
# Global Parameters
# ---------------------------
n_restarts = 5
training_percent = 0.5

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

def train_config(n,percent):
    global n_restarts
    global training_percent
    n_restarts = n
    training_percent = percent
    print(f'n_restarts: {n_restarts}, training_percent: {training_percent}')
    pass

def train_gp(X_train, y_train):
    start_time = time.time()
    gp = gp_emulator.GaussianProcess(inputs = X_train, targets = y_train)
    gp.learn_hyperparameters(n_tries = n_restarts, verbose = False)
    print('gp_emulator done', time.time() - start_time)
    return gp

def train_scikit(X_train, y_train, kernel):
    start_time = time.time()
    gp_scikit = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = n_restarts)
    gp_scikit.fit(X_train, y_train)
    print('scikit done', time.time() - start_time)
    return gp_scikit


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

###############################################################################################
"""
Use 20 percent of the grid points for training
Only a single scalar for length_scale parameter
Old scripts
"""
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


###############################################################################################

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

######################################################################################

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


# %%
# Training Setup
# --------------

def train_20_vector():
    """
    Use 20 percent of the grid points for training
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """

    # Setting number of restarts and training portion
    pass

def train_50():
    """
    Use 50 percent of the grid points for training
    Only a single scalar for length_scale parameter
    """
    # Setting number of restarts and training portion

    pass

def train_50_vector():
    """
    Use 50 percent of the grid points for training
    Use a vector length_scale -- training a length_scale parameter
    for each dimension of the input X
    """
    # Setting number of restarts and training portion

def cross_validation_5fold():
    """
    5-fold cross validation
    Use a single scalar for length_scale parameter
    """
    pass

def cross_validation_5fold_vector():
    """
    5-fold cross validation
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    pass


######################################################################################
# Even older stuffs
from model_sim import *
print('import successful')

import numpy as np
import matplotlib.pyplot as plt

import gp_emulator
from operator import itemgetter

##########################################################
########            Example Start             ############
##########################################################

# np.random.seed(42)
# n_samples = 2000
# x = np.linspace(0, 2, n_samples)
# y = np.exp(-0.7*x)*np.sin(2*np.pi*x/0.9)
# y += np.random.randn(n_samples)*0.02

# # Select a few random samples from x and y
# isel = np.random.choice(n_samples, 100)
# print(isel)
# x_train = np.atleast_2d(x[isel]).T
# y_train = y[isel]
# fig = plt.figure(figsize=(12,4))

# gp = gp_emulator.GaussianProcess(x_train, y_train)
# gp.learn_hyperparameters(n_tries=10,verbose = False)

# y_pred, y_unc, _ = gp.predict(np.atleast_2d(x).T,
#                                 do_unc=True, do_deriv=False)
# plt.plot(x, y_pred, '-', lw=2., label="Predicted")
# plt.plot(x, np.exp(-0.7*x)*np.sin(2*np.pi*x/0.9), '-', label="True")
# plt.fill_between(x, y_pred-1.96*y_unc,y_pred+1.96*y_unc, color="0.8")
# plt.legend(loc="best")
# plt.show()

##########################################################
########            qRW_Newton                ############
##########################################################

# n_samples = 200
# phi = np.linspace(0, 1, n_samples)
# y = qRW_Newton(0.9,phi,1,50)



# isel = np.random.choice(n_samples,50)
# phi_train = np.atleast_2d(phi[isel]).T
# y_train = y[isel]

# gp = gp_emulator.GaussianProcess(phi_train, y_train)
# gp.learn_hyperparameters(n_tries = 5)
# y_pred, y_unc, _ = gp.predict(np.atleast_2d(phi).T, do_unc = True, do_deriv=False)

# fig = plt.figure(figsize=(12,4))
# plt.plot(phi,y,'-',lw=8.,label='True')
# plt.plot(phi,y_pred,'-',lw=2.,label="Predicted")
# plt.legend(loc="best")
# plt.show()

##########################################################
########      higher Dim input                ############
##########################################################

np.random.seed(42)
n_samples = 2000
# x = np.linspace(0, 2, n_samples)
x1 = np.linspace(0,2, n_samples)
x2 = np.linspace(2,4,n_samples)
# # y = np.exp(-0.7*x)*np.sin(2*np.pi*x/0.9)
y = np.exp(-0.7*x1)*np.sin(2*np.pi*x1/0.9) + x2
y += np.random.randn(n_samples)*0.02

x = np.stack((x1,x2),axis=-1)
x_1, x_2 = itemgetter(*[0,1])(np.split(x,2,1))

# # Select a few random samples from x and y
isel = np.random.choice(n_samples, 500)
# print(isel)
# isel = np.linspace(0,9,10,dtype=int)
x_train = x[isel]
y_train = y[isel]

gp = gp_emulator.GaussianProcess(x_train, y_train)
gp.learn_hyperparameters(n_tries=10,verbose = False)
y_pred, y_unc, _ = gp.predict(x,do_unc=True, do_deriv=False)

fig = plt.figure(figsize=(12,4))
plt.plot(x_1, y_pred, '-', lw=2., label="Predicted")
plt.plot(x_1, np.exp(-0.7*x1)*np.sin(2*np.pi*x1/0.9)+x2, '-', label="True")
plt.plot(x_1,y,'-', label="True")
# plt.plot(x_2, np.exp(-0.7*x1)*np.sin(2*np.pi*x1/0.9)+x2, '-', label="True")
# plt.fill_between(x, y_pred-1.96*y_unc,y_pred+1.96*y_unc, color="0.8")
plt.legend(loc="best")
plt.show()

# Converting two vectors to a coordinate grid
# def coord(x1, x2):
#     x1x1, x2x2 = np.meshgrid(x1,x2)
#     coordinates = np.array((x1x1.ravel(),x2x2.ravel())).T
#     return coordinates

# Converts n 1-Dim vectors (of length v1,v2,...,vn) to n-Dim grid
# then, return the coordinates, which is ndarray of shape (v1*v2*...*vn, n)
def NDimCoord(*args):
    return np.vstack(np.meshgrid(*args)).reshape((len(args),-1)).T

b1 = np.array([1,2,3])
b2 = np.array([4,5])
b3 = np.array([6,7,8])
mycoord = NDimCoord(b1,b2,b3)
x = np.delete(mycoord,-1,axis=1) # shape of (N,m)
y = mycoord[:,-1] # shape of (N,)


########################################################################
##################### Test Field  ######################################
########################################################################

# x1 = np.array([1,2,3])
# x2 = np.array([4,5,6])
# x3 = np.array([7,8,9])
# x1x1x1,x2x2x2,x3x3x3 = np.meshgrid(x1,x2,x3,indexing = 'ij')
# x1x1x1[:,0,0]
# x2x2x2[0,:,0]
# x3x3x3[0,0,:]
# x1x1x1,x2x2x2,x3x3x3 = np.meshgrid(x1,x2,x3,indexing = 'xy')

# np.arange(2)
# np.arange(3)
# A,B = np.meshgrid(np.arange(2),np.arange(3))
# A+B

# np.vstack(np.meshgrid(x1,x2,x3)).reshape(3,-1).T

# a1 = np.array([1,2,3])
# a2 = np.array([4,5,6])
# a3 = np.array([7,8,9])
# a4 = np.array([10,11,12])
# np.vstack(np.meshgrid(a1,a2,a3,a4)).reshape(shape=(4,-1))
# stack = np.vstack(np.meshgrid(a1,a2,a3,a4))
# stack.reshape((4,-1)).T # -1 denotes inferred length
# # cant use shape = (4,-1)
# # only `order` is keyword argument; treat shape as positional argument; 
# stack.reshape(shape=(4,-1))



# np.vstack(np.meshgrid(a1,a2,a3,a4)).reshape(4,-1).T

def f(x, y):
    return x*y

a = np.linspace(1, 10, 10)
b = np.linspace(-10, -1, 10)

A, B = np.meshgrid(a, b)
C = f(A, B)

########
# Plot #
########

# # 2D plot
# fig = plt.figure(figsize=(12,4))
# plt.plot(phi_coor, y_pred, '-', lw=2., label = "Predicted")
# plt.plot(phi_coor, y, '-', label = "True")
# plt.legend(loc="best")
# plt.show()

# # 3D plot
# X = phi_coor.ravel()
# Y = gamma_coor.ravel()
# fig = plt.figure()
# ax = plt.axes(projection = '3d')
# ax.plot_trisurf(X,Y,y_pred,color='orange') # Predicted
# ax.plot_trisurf(X,Y,y.ravel(),color='blue') # True

# ax.plot_trisurf(X,Y,y_pred2,color='orange') # Predicted
# ax.plot_trisurf(X,Y,y2,color='blue') # True
# # ax.plot_surface(phi_coor.ravel(),gamma_coor.ravel(),y.ravel())
# # ax.scatter(phi_coor,gamma_coor,y,s=5,c='b',marker='o')
# # ax.scatter(phi_coor,gamma_coor,y_pred,s=5,c='r',marker='^')
# ax.set_title('surface')
# ax.set_xlabel('phi')
# ax.set_ylabel('gamma')
# ax.set_zlabel('inverse')
# plt.show()

## July 14 #######################################################################
# %%
# imports
# -------
from model_sim import *

import math
import time
import os
from operator import itemgetter
from distutils.util import strtobool

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# import gp_emulator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern
from sklearn.model_selection import KFold

# %%
# Configure Training Settings
# Global Parameters
# ---------------------------
# n_restarts = 5
# training_percent = 0.5

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

def pack_MSE(type, gaussian_process, X, X_test, X_train, y, y_test, y_train):
    """
    return a list [overall, test, train] of MSE values
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

def train_config(n,percent):
    """
    Modify the global parameters n_restarts and portion used for training
    """
    global n_restarts
    global training_percent
    n_restarts = n
    training_percent = percent
    print(f'n_restarts: {n_restarts}, training_percent: {training_percent}')
    pass

# def train_gp(X_train, y_train):
#     start_time = time.time()
#     gp = gp_emulator.GaussianProcess(inputs = X_train, targets = y_train)
#     gp.learn_hyperparameters(n_tries = n_restarts, verbose = False)
#     print('gp_emulator done', time.time() - start_time)
#     return gp

def train_scikit(X_train, y_train, kernel):
    start_time = time.time()
    print('n_restarts:', n_restarts)
    gp_scikit = GaussianProcessRegressor(kernel = kernel, 
                                        n_restarts_optimizer = n_restarts,
                                        copy_X_train = False)
    gp_scikit.fit(X_train, y_train)
    print('scikit done', time.time() - start_time)
    return gp_scikit

def train(data_exist: bool = False, y_exist: bool = False, portion: float = 0.2, restarts: int = 10):
    return

# %%
# Train_20s
# ---------------------------------------------------------------------------------------------------
def train_20(data_exist: bool = False, save_emulator = True):
    """
    Use 20 percent of the grid points for training 
    X = grids of (\phi, \gamma), p = 0.9 held constant
    Only a single scalar for length_scale parameter
    Train Scikit GPs using RBF, RQ, and Matern v = 1/2, 3/2, and 5/2 kernels
    """
    print('running train_20 ...')
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.2
    setup = 'train_20/'
    path = './data/' + setup

    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        phis = np.linspace(0,1,100) # shape = (v1,)
        gammas = np.linspace(1,2,100) # shape = (v2,)
        X = NDimCoord(phis,gammas) # create a grid of the params
        phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    # Train Emulators (Scikit models)
    gp_scikit_RBF = train_scikit(X_train, y_train, RBF())
    gp_scikit_RQ = train_scikit(X_train, y_train, RationalQuadratic())
    gp_scikit_M1 = train_scikit(X_train, y_train, Matern(nu=0.5))
    gp_scikit_M2 = train_scikit(X_train ,y_train, Matern(nu=1.5))
    gp_scikit_M3 = train_scikit(X_train, y_train, Matern(nu=2.5))

    if save_emulator == True:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_20 completed.')
    pass

def train_20_vector(data_exist = False, save_emulator = True):
    """
    Use 20 percent of the grid points for training
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    print('running train_20_vector ...')
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.2
    setup = 'train_20_vector/'
    path = './data/' + setup
    
    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        phis = np.linspace(0,1,100) # shape = (v1,)
        gammas = np.linspace(1,2,100) # shape = (v2,)
        X = NDimCoord(phis,gammas) # create a grid of the params
        phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=0.5: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=2.5: ', end_time - start_time)


    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_20_vector completed.')
    pass

def train_20_vector_full(data_exist = False, save_emulator = True):
    """
    Use 20 percent of the grid points for training
    X = (p, phi, gamma)
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    print('running train_20_vector_full ...')
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.2
    setup = 'train_20_vector_full/'
    path = './data/' + setup
    
    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        ps = np.linspace(0.8,0.9999,100)
        phis = np.linspace(1e-10,1,100) # shape = (v1,)
        gammas = np.linspace(0.5,2,100) # shape = (v2,)
        X = NDimCoord(ps,phis,gammas) # create a grid of the params
        p_coor, phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        start_time = time.time()
        y = qRW_Newton(p_coor, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        end_time = time.time()
        print('y calculated.', end_time - start_time)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'p_coor', p_coor)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        p_coor = np.load(path + 'p_coor.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    # Train emulators (Scikit Models)

    # RBF
    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)
    
    # RQ
    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)
    
    # Matern v = 0.5
    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0,1.0]),nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern: ', end_time - start_time)
    
    # Matern v = 1.5
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0,1.0]),nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)
    
    # Matern v = 2.5
    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0,1.0]),nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v = 2.5: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_20_vector_full completed.')
    pass

def train_20_vector_full_RBF(data_exist = False, save_emulator = True):
    """
    Use 20 percent of the grid points for training
    X = (p, phi, gamma)
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    print('running train_20_vector_full_RBF ...')
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.2
    setup = 'train_20_vector_full/'
    path = './data/' + setup
    
    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        ps = np.linspace(0.8,0.9999,100)
        phis = np.linspace(1e-10,1,100) # shape = (v1,)
        gammas = np.linspace(0.5,2,100) # shape = (v2,)
        X = NDimCoord(ps,phis,gammas) # create a grid of the params
        p_coor, phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        start_time = time.time()
        y = qRW_Newton(p_coor, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        end_time = time.time()
        print('y calculated.', end_time - start_time)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'p_coor', p_coor)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        p_coor = np.load(path + 'p_coor.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    # Train emulators (Scikit Models)

    # RBF
    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)

    print({'RBF_MSE': RBF_MSE})
    print('train_20_vector_full_RBF completed.')
    pass

# %%
# train_20_vector_split_RBF
# -------------------------
def train_20_vector_split_RBF(data_exist = False, save_emulator = True):
    """
    Use 20 percent of the grid points for training
    X = (p, phi, gamma)
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    print('running train_20_vector_split_RBF ...') # edit
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.2 # edit
    setup = 'train_20_vector_split/' # edit
    path = './data/' + setup
    
    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        ps = np.linspace(0.9,0.9999,100)[0:10]
        phis = np.linspace(0,1,100) # shape = (v1,)
        gammas = np.linspace(1,2,100) # shape = (v2,)
        X = NDimCoord(ps,phis,gammas) # create a grid of the params
        p_coor, phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        start_time = time.time()
        y = qRW_Newton(p_coor, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        end_time = time.time()
        print('y calculated.', end_time - start_time)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'p_coor', p_coor)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        p_coor = np.load(path + 'p_coor.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    # Train emulators (Scikit Models)

    # RBF
    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)

    print({'RBF_MSE': RBF_MSE})
    print('train_20_vector_split_RBF completed.')
    pass

# %%
# train_20_alpha
# --------------
def train_20_alpha(data_exist = False, save_emulator = True):
    print('running train_20_alpha...')
    setup = 'train_20_alpha/'
    path = './data/' + setup
    global n_restarts
    global training_percent
    n_restarts = 10
    training_percent = 0.2

    if data_exist == False:
        print('Generating data...\n')
        phis = np.linspace(0,1,100)
        gammas = np.linspace(1,2,100)
        X = NDimCoord(phis,gammas)
        phi_coor, gamma_coor = np.split(X,X.shape[1],1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]
        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else:
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(),
                                            n_restarts_optimizer = n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=0.5: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=2.5: ', end_time - start_time)

    if save_emulator:
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)
    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)
    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_20_alpha completed.')

# %%
# train_20_vector_alpha
# ---------------------
def train_20_vector_alpha(data_exist = False, save_emulator = True):
    print('running train_20_vector_alpha ...')
    setup = 'train_20_vector_alpha/'
    path = './data/' + setup
    global n_restarts
    global training_percent
    n_restarts = 10
    training_percent = 0.2

    if data_exist == False:
        print('Generating data...\n')
        phis = np.linspace(0,1,100)
        gammas = np.linspace(1,2,100)
        X = NDimCoord(phis,gammas)
        phi_coor, gamma_coor = np.split(X,X.shape[1],1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]
        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else:
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=0.5: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=2.5: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_20_vector_alpha completed.')


# %%
# train_20_vector_full_alpha
# --------------------------

def train_20_vector_full_alpha(data_exist = False, save_emulator = True):
    pass

# %%
# Train_50s
# ---------------------------------------------------------------------------------------------------

def train_50(data_exist = False, save_emulator = True):
    """
    Use 50 percent of the grid points for training
    Only a single scalar for length_scale parameter
    """
    print('running train_50 ...')
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.5
    setup = 'train_50/'
    path = './data/' + setup

    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        phis = np.linspace(0,1,100) # shape = (v1,)
        gammas = np.linspace(1,2,100) # shape = (v2,)
        X = NDimCoord(phis,gammas) # create a grid of the params
        phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    # Train Emulators (Scikit models)
    gp_scikit_RBF = train_scikit(X_train, y_train, RBF())
    gp_scikit_RQ = train_scikit(X_train, y_train, RationalQuadratic())
    gp_scikit_M1 = train_scikit(X_train, y_train, Matern(nu=0.5))
    gp_scikit_M2 = train_scikit(X_train ,y_train, Matern(nu=1.5))
    gp_scikit_M3 = train_scikit(X_train, y_train, Matern(nu=2.5))

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_50 completed.')
    pass

def train_50_vector(data_exist = False, save_emulator = True):
    """
    Use 50 percent of the grid points for training
    Use a vector length_scale -- training a length_scale parameter
    for each dimension of the input X
    """
    print('running train_50_vector ...')
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.5
    setup = 'train_50_vector/'
    path = './data/' + setup
    
    # Generate/Load Dataset
    if data_exist == False: # data_exist is False
        ## Generation
        print('Generating data...\n')
        phis = np.linspace(0,1,100) # shape = (v1,)
        gammas = np.linspace(1,2,100) # shape = (v2,)
        X = NDimCoord(phis,gammas) # create a grid of the params
        phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]

        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else: # data_exist is True
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=0.5: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=2.5: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_50_vector completed.')
    pass

def train_50_vector_full(data_exist = False, y_exist = False, save_emulator = True):
    """
    Use 50 percent of the grid points for training
    X = (p, phi, gamma)
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    print('running train_50_vector_full ...') # Edit
    # Setting Parameters
    global n_restarts
    n_restarts = 10
    global training_percent
    training_percent = 0.5 # Edit
    setup = 'train_50_vector_full/' # Edit
    path = './data/' + setup
    
    # Generate/Load Dataset
    if data_exist == True: # all the data are available
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        p_coor = np.load(path + 'p_coor.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')
    else: # data_exist is False
        if y_exist == True: # y is already calculated
            print('Reading y from ' + path + 'directory and partition training and testing set')
            # Loading X and y
            X = np.load(path + 'X.npy')
            y = np.load(path + 'y.npy')
            p_coor = np.load(path + 'p_coor.npy')
            phi_coor = np.load(path + 'phi_coor.npy')
            gamma_coor = np.load(path + 'gamma_coor.npy')
            # Partition training and testing set
            np.random.seed(42)
            n_samples = y.size
            training_size = math.floor(n_samples * training_percent)
            training_indices = np.random.choice(n_samples,training_size,replace = False)
            testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
            X_train, y_train = X[training_indices], y[training_indices]
            X_test, y_test = X[testing_indices], y[testing_indices]
            # Save the partitioned training and testing dataset
            np.save(path + 'X_train', X_train)
            np.save(path + 'X_test', X_test)
            np.save(path + 'y_train', y_train)
            np.save(path + 'y_test', y_test)
            print('Partition finished and saved under ' + path + 'directory.')
        else: # no data at all; need to calculate everything
            ## Generation
            print('Generating data...\n')
            ps = np.linspace(0.8,0.9999,100)
            phis = np.linspace(1e-10,1,100) # shape = (v1,)
            gammas = np.linspace(0.5,2,100) # shape = (v2,)
            X = NDimCoord(ps,phis,gammas) # create a grid of the params
            p_coor, phi_coor, gamma_coor = np.split(X,X.shape[1],1) # each shape = (v1*v2, 1)
            start_time = time.time()
            y = qRW_Newton(p_coor, phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
            end_time = time.time()
            print('y calculated.', end_time - start_time)
            y = y.ravel()
            np.random.seed(42)
            n_samples = y.size
            training_size = math.floor(n_samples * training_percent)
            training_indices = np.random.choice(n_samples,training_size,replace = False)
            testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
            X_train, y_train = X[training_indices], y[training_indices]
            X_test, y_test = X[testing_indices], y[testing_indices]

            ## Save Files
            os.makedirs(path, exist_ok = True)
            np.save(path + 'X', X)
            np.save(path + 'y', y)
            np.save(path + 'p_coor', p_coor)
            np.save(path + 'phi_coor', phi_coor)
            np.save(path + 'gamma_coor', gamma_coor)
            np.save(path + 'X_train', X_train)
            np.save(path + 'X_test', X_test)
            np.save(path + 'y_train', y_train)
            np.save(path + 'y_test', y_test)
            print('Data generation completed. Saved under the ' + path + ' directory.')
    # Train emulators (Scikit Models)
    print('start training...')
    # RBF
    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)
    
    # RQ
    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)
    
    # Matern v = 0.5
    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0,1.0]),nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern: ', end_time - start_time)
    
    # Matern v = 1.5
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0,1.0]),nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)
    
    # Matern v = 2.5
    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0,1.0]),nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v = 2.5: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_20_vector_full completed.')
    pass

def train_50_vector_full_alpha(data_exist = False, save_emulator = True):
    pass

# %%
# train_50_alpha
# --------------
def train_50_alpha(data_exist = False, save_emulator = True):
    print('running train_50_alpha...')
    setup = 'train_50_alpha/'
    path = './data/' + setup
    global n_restarts
    global training_percent
    n_restarts = 10
    training_percent = 0.5

    if data_exist == False:
        print('Generating data...\n')
        phis = np.linspace(0,1,100)
        gammas = np.linspace(1,2,100)
        X = NDimCoord(phis,gammas)
        phi_coor, gamma_coor = np.split(X,X.shape[1],1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]
        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else:
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(),
                                            n_restarts_optimizer = n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=0.5: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)

    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=2.5: ', end_time - start_time)

    if save_emulator:
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)
    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)
    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_50_alpha completed.')

# %%
# train_50_vector_alpha
# ---------------------
def train_50_vector_alpha(data_exist = False, save_emulator = True):
    print('running train_50_vector_alpha ...')
    setup = 'train_50_vector_alpha/'
    path = './data/' + setup
    global n_restarts
    global training_percent
    n_restarts = 10
    training_percent = 0.5

    if data_exist == False:
        print('Generating data...\n')
        phis = np.linspace(0,1,100)
        gammas = np.linspace(1,2,100)
        X = NDimCoord(phis,gammas)
        phi_coor, gamma_coor = np.split(X,X.shape[1],1)
        y = qRW_Newton(0.9, phi_coor, gamma_coor, 50)
        y = y.ravel()
        np.random.seed(42)
        n_samples = y.size
        training_size = math.floor(n_samples * training_percent)
        training_indices = np.random.choice(n_samples,training_size,replace = False)
        testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
        X_train, y_train = X[training_indices], y[training_indices]
        X_test, y_test = X[testing_indices], y[testing_indices]
        ## Save Files
        os.makedirs(path, exist_ok = True)
        np.save(path + 'X', X)
        np.save(path + 'y', y)
        np.save(path + 'phi_coor', phi_coor)
        np.save(path + 'gamma_coor', gamma_coor)
        np.save(path + 'X_train', X_train)
        np.save(path + 'X_test', X_test)
        np.save(path + 'y_train', y_train)
        np.save(path + 'y_test', y_test)
        print('Data generation completed. Saved under the ' + path + ' directory.')
    else:
        # Load Dataset
        print('Reading from generated data under the ' + path + ' directory.')
        X = np.load(path + 'X.npy')
        y = np.load(path + 'y.npy')
        phi_coor = np.load(path + 'phi_coor.npy')
        gamma_coor = np.load(path + 'gamma_coor.npy')
        X_train = np.load(path + 'X_train.npy')
        X_test = np.load(path + 'X_test.npy')
        y_train = np.load(path + 'y_train.npy')
        y_test = np.load(path + 'y_test.npy')

    start_time = time.time()
    gp_scikit_RBF = GaussianProcessRegressor(kernel = RBF(length_scale=np.array([1.0,1.0])),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train, y_train)
    end_time = time.time()
    print('RBF: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_RQ = GaussianProcessRegressor(kernel = RationalQuadratic(),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('RQ: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M1 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=0.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=0.5: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M2 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=1.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=1.5: ', end_time - start_time)
    
    start_time = time.time()
    gp_scikit_M3 = GaussianProcessRegressor(kernel=Matern(length_scale=np.array([1.0,1.0]),nu=2.5),
                                            n_restarts_optimizer=n_restarts,
                                            copy_X_train=False,
                                            alpha=0).fit(X_train,y_train)
    end_time = time.time()
    print('Matern v=2.5: ', end_time - start_time)

    if save_emulator:
        ## Save Emulators
        save_scikit_emulator(path + 'gp_scikit_RBF', gp_scikit_RBF)
        save_scikit_emulator(path + 'gp_scikit_RQ', gp_scikit_RQ)
        save_scikit_emulator(path + 'gp_scikit_M1', gp_scikit_M1)
        save_scikit_emulator(path + 'gp_scikit_M2', gp_scikit_M2)
        save_scikit_emulator(path + 'gp_scikit_M3', gp_scikit_M3)

    # Evaluations
    RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
    RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
    M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
    M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
    M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)

    np.save(path + 'RBF_MSE', RBF_MSE)
    np.save(path + 'RQ_MSE', RQ_MSE)
    np.save(path + 'M1_MSE', M1_MSE)
    np.save(path + 'M2_MSE', M2_MSE)
    np.save(path + 'M3_MSE', M3_MSE)

    print({'RBF_MSE': RBF_MSE, 'RQ_MSE': RQ_MSE, 
            'M1_MSE': M1_MSE, 'M2_MSE': M2_MSE, 'M3_MSE': M3_MSE})
    print('train_50_vector_alpha completed.')


# %%
# Cross Validations
# ---------------------------------------------------------------------------------------------------


def cross_validation_5fold():
    """
    5-fold cross validation
    Use a single scalar for length_scale parameter
    """
    pass

def cross_validation_5fold_vector():
    """
    5-fold cross validation
    Use a vector length_scale -- training a length scale parameter
    for each dimension of the input X
    """
    pass

def cross_validation_5fold_vector_full():
    pass

def cross_validation_5fold_vector_full_alpha():
    pass


# %%
# Tests and Scratch
# --------------------------------------------------------------------------------------------------
def test_param(data_exist = False, save_emulator = True):
    print('data_exist: ', data_exist)
    print('data_exist == True: ', data_exist==True)
    print('save_emulator: ', save_emulator)
    print('save_emulator == True: ', save_emulator == True)
    print('type(data_exist)', type(data_exist))

# %%
# main
# -----
if __name__ == '__main__':
    args = sys.argv
    if not args[2:]: # args[2:] is empty
        globals()[args[1]]()
    else: # args[2:] is not empty
        globals()[args[1]](*[strtobool(arg) for arg in args[2:]])