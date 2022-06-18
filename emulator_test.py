from model_sim import *
print('import successful')

import numpy as np
import matplotlib.pyplot as plt

import gp_emulator
from operator import itemgetter

##########################################################
########            Example Start                ############
##########################################################

# np.random.seed(42)
# n_samples = 2000
# x = np.linspace(0, 2, n_samples)
# y = np.exp(-0.7*x)*np.sin(2*np.pi*x/0.9)
# y += np.random.randn(n_samples)*0.02

# # Select a few random samples from x and y
# isel = np.random.choice(n_samples, 100)
# print(isel)
# x_train = np.atleast_2d(x[isel]).T
# y_train = y[isel]
# fig = plt.figure(figsize=(12,4))

# gp = gp_emulator.GaussianProcess(x_train, y_train)
# gp.learn_hyperparameters(n_tries=10,verbose = False)

# y_pred, y_unc, _ = gp.predict(np.atleast_2d(x).T,
#                                 do_unc=True, do_deriv=False)
# plt.plot(x, y_pred, '-', lw=2., label="Predicted")
# plt.plot(x, np.exp(-0.7*x)*np.sin(2*np.pi*x/0.9), '-', label="True")
# plt.fill_between(x, y_pred-1.96*y_unc,y_pred+1.96*y_unc, color="0.8")
# plt.legend(loc="best")
# plt.show()

##########################################################
########            qRW_Newton                ############
##########################################################

# n_samples = 200
# phi = np.linspace(0, 1, n_samples)
# y = qRW_Newton(0.9,phi,1,50)



# isel = np.random.choice(n_samples,50)
# phi_train = np.atleast_2d(phi[isel]).T
# y_train = y[isel]

# gp = gp_emulator.GaussianProcess(phi_train, y_train)
# gp.learn_hyperparameters(n_tries = 5)
# y_pred, y_unc, _ = gp.predict(np.atleast_2d(phi).T, do_unc = True, do_deriv=False)

# fig = plt.figure(figsize=(12,4))
# plt.plot(phi,y,'-',lw=8.,label='True')
# plt.plot(phi,y_pred,'-',lw=2.,label="Predicted")
# plt.legend(loc="best")
# plt.show()

##########################################################
########      higher Dim input                ############
##########################################################

np.random.seed(42)
n_samples = 2000
# x = np.linspace(0, 2, n_samples)
x1 = np.linspace(0,2, n_samples)
x2 = np.linspace(2,4,n_samples)
# # y = np.exp(-0.7*x)*np.sin(2*np.pi*x/0.9)
y = np.exp(-0.7*x1)*np.sin(2*np.pi*x1/0.9) + x2
y += np.random.randn(n_samples)*0.02

x = np.stack((x1,x2),axis=-1)
x_1, x_2 = itemgetter(*[0,1])(np.split(x,2,1))

# # Select a few random samples from x and y
isel = np.random.choice(n_samples, 500)
# print(isel)
# isel = np.linspace(0,9,10,dtype=int)
x_train = x[isel]
y_train = y[isel]

gp = gp_emulator.GaussianProcess(x_train, y_train)
gp.learn_hyperparameters(n_tries=10,verbose = False)
y_pred, y_unc, _ = gp.predict(x,do_unc=True, do_deriv=False)

fig = plt.figure(figsize=(12,4))
plt.plot(x_1, y_pred, '-', lw=2., label="Predicted")
plt.plot(x_1, np.exp(-0.7*x1)*np.sin(2*np.pi*x1/0.9)+x2, '-', label="True")
plt.plot(x_1,y,'-', label="True")
# plt.plot(x_2, np.exp(-0.7*x1)*np.sin(2*np.pi*x1/0.9)+x2, '-', label="True")
# plt.fill_between(x, y_pred-1.96*y_unc,y_pred+1.96*y_unc, color="0.8")
plt.legend(loc="best")
plt.show()

# Converting two vectors to a coordinate grid
# def coord(x1, x2):
#     x1x1, x2x2 = np.meshgrid(x1,x2)
#     coordinates = np.array((x1x1.ravel(),x2x2.ravel())).T
#     return coordinates

# Converts n 1-Dim vectors (of length v1,v2,...,vn) to n-Dim grid
# then, return the coordinates, which is ndarray of shape (v1*v2*...*vn, n)
def NDimCoord(*args):
    return np.vstack(np.meshgrid(*args)).reshape((len(args),-1)).T

b1 = np.array([1,2,3])
b2 = np.array([4,5])
b3 = np.array([6,7,8])
mycoord = NDimCoord(b1,b2,b3)
x = np.delete(mycoord,-1,axis=1) # shape of (N,m)
y = mycoord[:,-1] # shape of (N,)


########################################################################
##################### Test Field  ######################################
########################################################################

# x1 = np.array([1,2,3])
# x2 = np.array([4,5,6])
# x3 = np.array([7,8,9])
# x1x1x1,x2x2x2,x3x3x3 = np.meshgrid(x1,x2,x3,indexing = 'ij')
# x1x1x1[:,0,0]
# x2x2x2[0,:,0]
# x3x3x3[0,0,:]
# x1x1x1,x2x2x2,x3x3x3 = np.meshgrid(x1,x2,x3,indexing = 'xy')

# np.arange(2)
# np.arange(3)
# A,B = np.meshgrid(np.arange(2),np.arange(3))
# A+B

# np.vstack(np.meshgrid(x1,x2,x3)).reshape(3,-1).T

# a1 = np.array([1,2,3])
# a2 = np.array([4,5,6])
# a3 = np.array([7,8,9])
# a4 = np.array([10,11,12])
# np.vstack(np.meshgrid(a1,a2,a3,a4)).reshape(shape=(4,-1))
# stack = np.vstack(np.meshgrid(a1,a2,a3,a4))
# stack.reshape((4,-1)).T # -1 denotes inferred length
# # cant use shape = (4,-1)
# # only `order` is keyword argument; treat shape as positional argument; 
# stack.reshape(shape=(4,-1))



# np.vstack(np.meshgrid(a1,a2,a3,a4)).reshape(4,-1).T

def f(x, y):
    return x*y

a = np.linspace(1, 10, 10)
b = np.linspace(-10, -1, 10)

A, B = np.meshgrid(a, b)
C = f(A, B)