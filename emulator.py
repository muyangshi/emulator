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