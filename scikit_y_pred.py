import math
import time
import os
import pickle

import numpy as np

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, DotProduct, Matern
from sklearn.model_selection import KFold

with open('./data/RBF_50_C/gp_scikit_RBF.pickle', 'rb') as f:
    gp = pickle.load(f)

p_test = np.linspace(0.8,0.9999,80)
phi_test = np.linspace(0,1,80) # shape(m1,)
gamma_test = np.linspace(0.5,2,80) # shape(m2,)
# p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
# X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
X_test = np.vstack(np.meshgrid(p_test,phi_test,gamma_test,indexing='ij')).reshape(3,-1).T
# y_test_matrix_ravel = np.load('./data/80x80x80_C/'+'y_test_matrix_ravel.npy') # shape(n1*n2*n3, )

# Unable to allocate 119 GiB of RAM
# start = time.time()
# y_pred = gp.predict(X_test, return_std = False)
# print('prediction of 80x80x80 used: ', round(time.time() - start, 2))

y_pred = []

for Xi in X_test:
    yi = gp.predict(Xi, return_std = False)
    y_pred.append(yi)
np.save('./data/RBF_50_C/y_pred_80x80x80',y_pred)