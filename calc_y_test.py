# %%
from model_sim import *
import numpy as np
import time
import os
import multiprocessing as mp
from multiprocessing import Pool
# %%
# Training Part

def quantile_wrapper_C(p,phi,gamma):
    return qRW_Newton(p,phi,gamma,100)

p_train = np.linspace(0.8,0.9999,2) # shape (n1,)
phi_train = np.linspace(0,1,10) # shape (n2, )
gamma_train = np.linspace(0.5,2,10) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)

for i in range(100,len(X_train)):
    print(i, qRW_Newton(X_train[i,0], X_train[i,1], X_train[i,2], 100))