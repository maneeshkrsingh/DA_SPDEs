from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from nudging.models.lineargaussian_model import LGModel
import os

os.makedirs('../../DA_Results/', exist_ok=True)

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
np.random.seed(12345)
nsteps = 1
xpoints = 41 # no of weather station

N_obs = 1
model = LGModel(100, nsteps, xpoints, seed = 12345,  scale = 1.0)
model.setup()
x, = SpatialCoordinate(model.mesh)

X_truth = model.allocate()
Q = np.random.normal(0, 0.5)
print('inital', Q)
u = X_truth[0]
u.assign(Q)


y_true = model.obs().dat.data[:]
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))


for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step
    y_true = model.obs().dat.data[:]
    y_true_full[i,:] = y_true
    #print(y_true)
    y_noise = model.rg.normal(0.0, 0.5)  

    y_obs = y_true + y_noise   
    y_obs_full[i,:] = y_obs 

print(y_noise, y_obs_full)
np.save("../../DA_Results/w_true.npy", y_true_full)
np.save("../../DA_Results/w_obs.npy", y_obs_full)


print(y_true_full.shape)
# plt.plot(y_true_full[:, 0], 'b-' , label='true soln')
# #plt.plot(y_true_full[:, 10],  label='bad soln')
# #plt.plot(y_obs_full[:, 10], 'r-',  label='noisy soln')
# plt.legend()
# plt.show()