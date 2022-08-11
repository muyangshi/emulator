# %%
# imports
# -------
from model_sim import *

import math
import time
import os
import pickle
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

# def save_scikit_emulator(name: str,obj):
#     np.savez(name,obj)

# def load_scikit_emulator(file):
#     loaded_npzfile = np.load(file, allow_pickle=True)
#     loaded_gaussian_process = loaded_npzfile['arr_0'][()]
#     return loaded_gaussian_process

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

# def train_scikit(X_train, y_train, kernel):
#     start_time = time.time()
#     print('n_restarts:', n_restarts)
#     gp_scikit = GaussianProcessRegressor(kernel = kernel, 
#                                         n_restarts_optimizer = n_restarts,
#                                         copy_X_train = False)
#     gp_scikit.fit(X_train, y_train)
#     print('scikit done', time.time() - start_time)
#     return gp_scikit

def load_all(path):
    """
    Assuming all the data are generated, 
    load all the necessary X and y for training GPs
    """
    X = np.load(path + 'X.npy')
    X_train = np.load(path + 'X_train.npy')
    X_test = np.load(path + 'X_test.npy')
    y = np.load(path + 'y.npy')
    y_train = np.load(path + 'y_train.npy')
    y_test = np.load(path + 'y_test.npy')
    return X, X_train, X_test, y, y_train, y_test

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

def generate_all(X_dim,training_proportion,path):
    # X
    ps = np.linspace(0.8,0.9999,100) if X_dim == 3 else None
    phis = np.linspace(1e-10,1,25) # shape = (v1,)
    gammas = np.linspace(0.5,2,25) # shape = (v2,)
    X = NDimCoord(ps,phis,gammas) if X_dim == 3 else NDimCoord(phis,gammas)
    if X_dim == 3:
        p_coor, phi_coor, gamma_coor = np.split(X,X.shape[1],1)
    else: # X_dim == 2
        phi_coor, gamma_coor = np.split(X,X.shape[1],1)
        p_coor = 0.9
    
    # y
    start_time = time.time()
    # y = qRW_Newton_py(p_coor, phi_coor, gamma_coor,400) # shape = (v1*v2*...*vn,1)
    y = qRW_Newton(p_coor, phi_coor, gamma_coor, 400) # C++ quantile function
    end_time = time.time()
    print('y calculated. Used: ', round(end_time - start_time, 2), ' seconds.')
    y = y.ravel()

    # Save Files
    os.makedirs(path, exist_ok = True)
    np.save(path + 'X', X)
    np.save(path + 'p_coor', p_coor)
    np.save(path + 'phi_coor', phi_coor)
    np.save(path + 'gamma_coor', gamma_coor)
    np.save(path + 'y', y)
    
    # X_train, X_test, y_train, y_test
    X_train, X_test, y_train, y_test = split_data(X,y,training_proportion,path)
    return X, X_train, X_test, y, y_train, y_test


# %%
# Training funtions
# -----------------
def train(name: str = 'No_Name', X_dim: int = 2, 
            y_exist: bool = False, data_exist: bool = False, 
            proportion: float = 0.2, restarts: int = 10):
    print('Running ' + name + '...')
    path = './data/' + name + '/'
    n_restarts = restarts
    training_proportion = proportion
    train_RBF, train_DP_RBF, train_RQ, train_M1, train_M2, train_M3 = True, True, False, False, False, False

    # Generate/Load Dataset
    print('\n##### Generate/Load Dataset #####\n')
    if y_exist == True:
        if data_exist == True:
            print('Read existing data from the ' + path + ' directory.')
            X, X_train, X_test, y, y_train, y_test = load_all(path)
        else:
            print('Reading y from ' + path + 'directory and partition training and testing set')
            X = np.load(path + 'X.npy')
            y = np.load(path + 'y.npy')
            X_train, X_test, y_train, y_test = split_data(X,y,training_proportion,path)
    else: # nothing exist
        print('No data exist. Start fresh calculation...')
        X, X_train, X_test, y, y_train, y_test = generate_all(X_dim, training_proportion, path)

    # Training
    np.random.seed(42)
    print('\n#### Training ####\n')
    if train_RBF == True:
        print('training with Radial Basis Function Kernel')
        start_time = time.time()
        gp_scikit_RBF = GaussianProcessRegressor(
                            kernel = RBF(length_scale=np.ones(X_dim),length_scale_bounds=(0.01,100000)),
                            n_restarts_optimizer=n_restarts,
                            copy_X_train=False,
                            normalize_y=True).fit(X_train, y_train)
        end_time = time.time()
        print('RBF: ', round(end_time - start_time,2),' seconds.')
        # RBF_MSE = pack_MSE('scikit',gp_scikit_RBF, X, X_test, X_train, y, y_test, y_train)
        # print(RBF_MSE)
        start_time = time.time()
        y_pred_all = gp_scikit_RBF.predict(X,return_std=False)
        print('RBF predict X: ', round(time.time() - start_time,2),' seconds')
        y_pred_test = gp_scikit_RBF.predict(X_test,return_std=False)
        y_pred_train = gp_scikit_RBF.predict(X_train,return_std=False)
        RBF_MSE = [MSE(y_pred_all, y),MSE(y_pred_test,y_test),MSE(y_pred_train,y_train)]
        print(RBF_MSE)
        np.save(path + 'RBF_MSE', RBF_MSE)
        np.save(path + 'y_pred_all',y_pred_all)
        np.save(path + 'y_pred_test',y_pred_test)
        np.save(path + 'y_pred_train',y_pred_train)
        with open(path+'gp_scikit_RBF.pickle','wb') as f:
            pickle.dump(gp_scikit_RBF,f,pickle.HIGHEST_PROTOCOL)
    if train_RQ == True:
        print('training with Rational Quadratic Kernel')
        start_time = time.time()
        gp_scikit_RQ = GaussianProcessRegressor(
                        kernel = RationalQuadratic(),
                        n_restarts_optimizer=n_restarts,
                        copy_X_train=False).fit(X_train,y_train)
        end_time = time.time()
        print('RQ: ', round(end_time - start_time,2),' seconds.')
        RQ_MSE = pack_MSE('scikit',gp_scikit_RQ, X, X_test, X_train, y, y_test, y_train)
        print(RQ_MSE)
        np.save(path + 'RQ_MSE', RQ_MSE)
    if train_M1 == True:
        print('training Matern kernel v = 0.5')
        start_time = time.time()
        gp_scikit_M1 = GaussianProcessRegressor(
                        kernel=Matern(nu=0.5,length_scale=np.ones(X_dim)),
                        n_restarts_optimizer=n_restarts,
                        copy_X_train=False).fit(X_train,y_train)
        end_time = time.time()
        print('Matern v = 0.5: ', round(end_time - start_time,2),' seconds.')
        M1_MSE = pack_MSE('scikit',gp_scikit_M1, X, X_test, X_train, y, y_test, y_train)
        print(M1_MSE)
        np.save(path + 'M1_MSE', M1_MSE)
    if train_M2 == True:
        print('training Matern Kernel v = 1.5')
        start_time = time.time()
        gp_scikit_M2 = GaussianProcessRegressor(
                        kernel=Matern(nu=1.5,length_scale=np.ones(X_dim)),
                        n_restarts_optimizer=n_restarts,
                        copy_X_train=False).fit(X_train,y_train)
        end_time = time.time()
        print('Matern v = 1.5: ', round(end_time - start_time,2),' seconds.')
        M2_MSE = pack_MSE('scikit',gp_scikit_M2, X, X_test, X_train, y, y_test, y_train)
        print(M2_MSE)
        np.save(path + 'M2_MSE', M2_MSE)
    if train_M3 == True:
        print('training Matern Kernel v = 2.5')
        start_time = time.time()
        gp_scikit_M3 = GaussianProcessRegressor(
                        kernel=Matern(nu=2.5,length_scale=np.ones(X_dim)),
                        n_restarts_optimizer=n_restarts,
                        copy_X_train=False).fit(X_train,y_train)
        end_time = time.time()
        print('Matern v = 2.5: ', round(end_time - start_time,2),' seconds.')
        M3_MSE = pack_MSE('scikit',gp_scikit_M3, X, X_test, X_train, y, y_test, y_train)
        print(M3_MSE)
        np.save(path + 'M3_MSE', M3_MSE)
    return

# %%
# main
# -----
if __name__ == '__main__':
    args = sys.argv
    if not args[2:]: # args[2:] is empty
        globals()[args[1]]()
    else: # args[2:] is not empty
        # globals()[args[1]](*[strtobool(arg) for arg in args[2:]])
        name = args[2]
        X_dim = int(args[3])
        y_exist = strtobool(args[4])
        data_exist = strtobool(args[5])
        portion = float(args[6])
        restarts = int(args[7])
        train(name, X_dim, y_exist, data_exist, portion, restarts)
        # print(args)