# from tempfile import TemporaryFile
from model_sim import *
from operator import itemgetter


import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import gp_emulator


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

##########################
# Generate training data #
##########################

# Specify parameter range
phis = np.linspace(0,1,100) # shape = (v1,)
gammas = np.linspace(1,2,100) # shape = (v2,)
# create grid of parameter 
x_grid = NDimCoord(phis,gammas) # shape = (v1*v2, 2)
phi_coor, gamma_coor = np.split(x_grid,x_grid.shape[1],1) # each shape = (v1*v2, 1)
# calculate y
y = qRW_Newton(0.9,phi_coor, gamma_coor, 50) # shape = (v1*v2*...*vn,1)
# data
data = np.hstack((x_grid,y))
np.save('data',data)
np.save('x_grid',x_grid)

############
# training #
############

np.random.seed(42)
n_samples = len(x_grid)
isel = np.random.choice(n_samples,100)
x_train = x_grid[isel]
y_train = y[isel].ravel() # the y was a vector
gp = gp_emulator.GaussianProcess(x_train,y_train)
gp.learn_hyperparameters(n_tries = 10, verbose = False)
y_pred, y_unc, _ = gp.predict(x_grid, do_unc = True, do_deriv = False)
np.save('y_pred',y_pred)

########
# Plot #
########

# 2D plot
fig = plt.figure(figsize=(12,4))
plt.plot(phi_coor, y_pred, '-', lw=2., label = "Predicted")
plt.plot(phi_coor, y, '-', label = "True")
plt.legend(loc="best")
plt.show()

# 3D plot
X = phi_coor.ravel()
Y = gamma_coor.ravel()
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_trisurf(X,Y,y_pred,color='orange') # Predicted
ax.plot_trisurf(X,Y,y.ravel(),color='blue') # True
# ax.plot_surface(phi_coor.ravel(),gamma_coor.ravel(),y.ravel())
# ax.scatter(phi_coor,gamma_coor,y,s=5,c='b',marker='o')
# ax.scatter(phi_coor,gamma_coor,y_pred,s=5,c='r',marker='^')
ax.set_title('surface')
ax.set_xlabel('phi')
ax.set_ylabel('gamma')
ax.set_zlabel('inverse')
plt.show()