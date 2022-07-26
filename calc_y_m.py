from model_sim import *
import numpy as np
import time
import os
from multiprocessing import Pool

# Training Part

def quantile_wrapper_C(p,phi,gamma):
    return qRW_Newton(p,phi,gamma,100)

p_train = np.linspace(0.8,0.9999,500) # shape (n1,)
phi_train = np.linspace(0,1,500) # shape (n2, )
gamma_train = np.linspace(0.5,2,500) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)
start = time.time()
pool = Pool(processes=96)
results = pool.starmap(quantile_wrapper_C,X_train)
y_train_matrix = np.array(results).reshape(500,500,500)
# y_train_matrix = qRW_Newton(p_train_g, phi_train_g, gamma_train_g, 100)
print('Calculation of y took: ', round(time.time() - start,2), ' seconds.')
y_train_matrix_ravel = y_train_matrix.ravel()
# y_train_vector = qRW_Newton(X_train_p, X_train_phi, X_train_gamma, 100)

path = './data/500x500x500/'
os.makedirs(path, exist_ok = True)
np.save(path+'y_train_matrix',y_train_matrix)
np.save(path+'y_train_matrix_ravel',y_train_matrix_ravel)
# np.save(path+'y_train_vector',y_train_vector) # y_train_vector.ravel() = y_train_matrix_ravel, using indexing='ij'


# Testing Part

# p_test = np.linspace(0.8,0.9999,80)
# phi_test = np.linspace(0,1,80) # shape(m1,)
# gamma_test = np.linspace(0.5,2,80) # shape(m2,)
# p_test_g, phi_test_g, gamma_test_g = np.meshgrid(p_test, phi_test, gamma_test, indexing = 'ij')
# X_test = np.vstack([p_test_g, phi_test_g, gamma_test_g]).reshape(3,-1).T # shape(n1*n2*n3, 3)
# # Do this, becuase the ordering the points would be consistent
# # with the test_pts, which will be input to the RegularGridInterpolator
# X_test_p, X_test_phi ,X_test_gamma = np.split(X_test,3,-1) # each shape (n_i, 1)
# y_test_matrix = qRW_Newton(p_test_g,phi_test_g, gamma_test_g, 100)
# y_test_matrix_ravel = y_test_matrix.ravel()
# # y_test_vector = qRW_Newton(X_test_p,X_test_phi,X_test_gamma,100)

# path = './data/80x80x80/'
# os.makedirs(path, exist_ok=True)
# np.save(path+'y_test_matrix',y_test_matrix)
# np.save(path+'y_test_matrix_ravel',y_test_matrix_ravel)
# # np.save(path+'y_test_vector',y_test_vector)