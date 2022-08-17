from model_sim import *

import time
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern

import GPy

from scipy.interpolate import RegularGridInterpolator

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