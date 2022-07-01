# %%
# imports
# -------
from model_sim import *
from operator import itemgetter
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import gp_emulator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern

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

# %%
# Splitting Dataset
# -----------------
np.random.seed(42)
n_samples = y.size
training_size = math.floor(n_samples * 0.2)
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
print(MSE(gp_y_pred_all,y))
print(MSE(gp_y_pred_test,y_test))
print(MSE(gp_y_pred_train,y_train))

# Scikit RBF
print('Scikit RBF')
print(MSE(gp_scikit_RBF_y_pred_all,y))
print(MSE(gp_scikit_RBF_y_pred_test,y_test))
print(MSE(gp_scikit_RBF_y_pred_train,y_train))

# Scikit RQ
print('Scikit Rational Quadratic')
print(MSE(gp_scikit_RQ_y_pred_all,y))
print(MSE(gp_scikit_RQ_y_pred_test,y_test))
print(MSE(gp_scikit_RQ_y_pred_train,y_train))

# Scikit Dot Product
print('Scikit Dot Product')
print(MSE(gp_scikit_DP_y_pred_all,y))
print(MSE(gp_scikit_DP_y_pred_test,y_test))
print(MSE(gp_scikit_DP_y_pred_train,y_train))

# Scikit Matern 1/2
print('Scikit Matern nu = 0.5')
print(MSE(gp_scikit_M1_y_pred_all,y))
print(MSE(gp_scikit_M1_y_pred_test,y_test))
print(MSE(gp_scikit_M1_y_pred_train,y_train))

# Scikit Matern 3/2
print('Scikit Matern nu = 1.5')
print(MSE(gp_scikit_M2_y_pred_all,y))
print(MSE(gp_scikit_M2_y_pred_test,y_test))
print(MSE(gp_scikit_M2_y_pred_train,y_train))

# Scikit Matern 5/2
print('Scikit Matern nu = 2.5')
print(MSE(gp_scikit_M3_y_pred_all,y))
print(MSE(gp_scikit_M3_y_pred_test,y_test))
print(MSE(gp_scikit_M3_y_pred_train,y_train))


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
# %%
