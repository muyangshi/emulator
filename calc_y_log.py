from model_sim import *
import numpy as np
import time
import os
from multiprocessing import Pool

# Training Part

def quantile_wrapper_C(p,phi,gamma):
    return qRW_Newton(p,phi,gamma,100)


p = np.geomspace(1,10,num=20)
y = [0] * 20
plt.plot(p,y,'bo')
plt.show()

import matplotlib.pyplot as plt
n = 10
p = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8,num=n)
y = [0] * n

for i in range(1,n):
    print((p[i]-(0.9999+0.8))/(p[i-1]-(0.9999+0.8)))

plt.plot(p,y,'bo')
plt.show()

p_train = (0.9999 + 0.8) + np.geomspace(-0.9999,-0.8, 500)
# phi_train = 





p_train = np.linspace(0.8,0.9999,50) # shape (n1,)
phi_train = np.linspace(0,1,50) # shape (n2, )
gamma_train = np.linspace(0.5,2,10) # shape (n3, )
p_train_g, phi_train_g, gamma_train_g = np.meshgrid(p_train, phi_train, gamma_train, indexing='ij')
X_train = np.vstack([p_train_g, phi_train_g, gamma_train_g]).reshape(3,-1).T # shape(n1*n2*n3,3)
X_train_p, X_train_phi, X_train_gamma = np.split(X_train,3,-1) # each shape (n_i, 1)
start = time.time()
pool = Pool(processes=6)
results = pool.starmap(quantile_wrapper_C,X_train)
y_train_matrix = np.array(results).reshape(50,50,10)
# y_train_matrix = qRW_Newton(p_train_g, phi_train_g, gamma_train_g, 100)
print('Calculation of y took: ', round(time.time() - start,2), ' seconds.')
y_train_matrix_ravel = y_train_matrix.ravel()
# y_train_vector = qRW_Newton(X_train_p, X_train_phi, X_train_gamma, 100)

path = './data/500x500x500/'
os.makedirs(path, exist_ok = True)
np.save(path+'y_train_matrix',y_train_matrix)
np.save(path+'y_train_matrix_ravel',y_train_matrix_ravel)
# np.save(path+'y_train_vector',y_train_vector) # y_train_vector.ravel() = y_train_matrix_ravel, using indexing='ij'
