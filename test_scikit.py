from model_sim import *

import math
import time
from operator import itemgetter

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

# import gp_emulator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern
from sklearn.model_selection import KFold

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

# def train_gp(X_train, y_train):
#     start_time = time.time()
#     gp = gp_emulator.GaussianProcess(inputs = X_train, targets = y_train)
#     gp.learn_hyperparameters(n_tries = 5, verbose = False)
#     print('gp_emulator done', time.time() - start_time)
#     return gp

def train_scikit(X_train, y_train, kernel):
    start_time = time.time()
    gp_scikit = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer = 10)
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

data = np.load('./data/data.npy')
y = np.load('./data/y.npy')
x_grid = np.load('./data/x_grid.npy')
phi_coor = np.load('./data/phi_coor.npy')
gamma_coor = np.load('./data/gamma_coor.npy')
X = x_grid

np.random.seed(42)
n_samples = y.size
training_size = math.floor(n_samples * 0.2)
training_indices = np.random.choice(n_samples,training_size,replace = False)
testing_indices = np.setdiff1d(np.arange(0,n_samples),training_indices)
X_train, y_train = x_grid[training_indices], y[training_indices]
X_test, y_test = x_grid[testing_indices], y[testing_indices]

gp_scikit_RBF = train_scikit(X_train, y_train, RBF())
RBF_kernel = RBF(length_scale=np.array([1,1]))
gp_scikit_RBF2 = train_scikit(X_train,y_train,RBF_kernel)
RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
print(RBF_MSE)

RBF_MSE2 = pack_MSE('scikit',gp_scikit_RBF2, X, X_test, X_train, y, y_test, y_train)