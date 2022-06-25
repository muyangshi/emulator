# from tempfile import TemporaryFile
from model_sim import *
from operator import itemgetter
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import gp_emulator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic

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
# save to npy files
data = np.hstack((x_grid,y))
np.save('data',data)
np.save('x_grid',x_grid)
np.save('y',y)

#########################
# training & prediction #
#########################

np.random.seed(42)
n_samples = len(x_grid)
isel = np.random.choice(n_samples,100)
x_train = x_grid[isel]
y_train = y[isel].ravel() # the y was a vector
gp = gp_emulator.GaussianProcess(inputs=x_train,targets=y_train)
gp.learn_hyperparameters(n_tries = 10, verbose = False)
y_pred, y_unc, _ = gp.predict(x_grid, do_unc = True, do_deriv = False)
np.save('y_pred',y_pred)
gp.save_emulator('gp') # saved as npz file

# Load trained emulator
gp2 = gp_emulator.GaussianProcess(emulator_file='gp.npz')
data2 = np.load('data.npy')
y2 = data2[:,-1]
x_grid2 = np.load('x_grid.npy')
y_pred2, y_unc2, _ = gp2.predict(x_grid, do_unc = False, do_deriv = False)

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

ax.plot_trisurf(X,Y,y_pred2,color='orange') # Predicted
ax.plot_trisurf(X,Y,y2,color='blue') # True
# ax.plot_surface(phi_coor.ravel(),gamma_coor.ravel(),y.ravel())
# ax.scatter(phi_coor,gamma_coor,y,s=5,c='b',marker='o')
# ax.scatter(phi_coor,gamma_coor,y_pred,s=5,c='r',marker='^')
ax.set_title('surface')
ax.set_xlabel('phi')
ax.set_ylabel('gamma')
ax.set_zlabel('inverse')
plt.show()

######################
# Using Scikit Learn #
######################

# rng = np.random.RandomState(42)
# training_indices = rng.choice(np.arange(y.size), size=100, replace=False)
np.random.seed(42)
n_samples = y.size
training_indices = np.random.choice(n_samples,1000,replace = False)
X_train, y_train = x_grid[training_indices], y[training_indices]

np.random.seed(42)
isel = np.random.choice(n_samples,size=100)
X_train, y_train = x_grid[isel],y[isel]

kernel = RBF()
gaussian_process = GaussianProcessRegressor(kernel=kernel,n_restarts_optimizer=10)
gaussian_process.fit(X_train,y_train)
gaussian_process.kernel_
mean_prediction, std_prediction = gaussian_process.predict(x_grid, return_std=True)

kernel2 = RationalQuadratic()
gaussian_process2 = GaussianProcessRegressor(kernel=kernel2,n_restarts_optimizer=10)
gaussian_process2.fit(X_train,y_train)
gaussian_process2.kernel_
mean_prediction2, std_prediction2 = gaussian_process2.predict(x_grid,return_std=True)

X = phi_coor.ravel()
Y = gamma_coor.ravel()
Z = mean_prediction
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_trisurf(X,Y,Z,color='orange') # Predicted
ax.plot_trisurf(X,Y,y.ravel(),color='blue') # True
ax.set_title('surface')
ax.set_xlabel('phi')
ax.set_ylabel('gamma')
ax.set_zlabel('inverse')
plt.show()

X = phi_coor.ravel()
Y = gamma_coor.ravel()
Z = mean_prediction2
fig = plt.figure()
ax = plt.axes(projection = '3d')
ax.plot_trisurf(X,Y,Z,color='orange') # Predicted
ax.plot_trisurf(X,Y,y.ravel(),color='blue') # True
ax.set_title('surface')
ax.set_xlabel('phi')
ax.set_ylabel('gamma')
ax.set_zlabel('inverse')
plt.show()

np.savez('gaussian_process',gaussian_process)
loaded_npzfile = np.load('gaussian_process.npz',allow_pickle=True)
loaded_gaussian_process = loaded_npzfile['arr_0'][()]
test = loaded_gaussian_process.predict(x_grid, return_std=False) # same!