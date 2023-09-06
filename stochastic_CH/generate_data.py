from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from nudging.models.stochastic_Camassa_Holm import Camsholm

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
N_obs = 1
nsteps = 5
xpoints = 40
model = Camsholm(100, nsteps, xpoints, lambdas=False)
model.setup()
obs_shape = model.obs().dat.data[:]
x, = SpatialCoordinate(model.mesh)

###################  To generate data only for obs data points 
X_truth = model.allocate()
_, u0 = X_truth[0].split()
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))



# save data at obs points
y_true = np.zeros((N_obs, np.size(obs_shape)))
y_obs = np.zeros((N_obs, np.size(obs_shape)))

for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step
    y = model.obs().dat.data[:]
    y_true[i,:] = y
    y_noise = np.random.normal(0.0, 0.025, xpoints)  
    y_obs[i,:] =  y + y_noise  
np.save("y_true.npy", y_true)
np.save("y_obs.npy", y_obs)


###################  To generate data for all time  data points 
Y_truth = model.allocate()
_, u0 = Y_truth[0].split()
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

# save data at all points
y_true_alltime = np.zeros((N_obs*nsteps, np.size(obs_shape)))
y_obs_alltime = np.zeros((N_obs*nsteps, np.size(obs_shape)))

for i in range(N_obs):
    z = []
    model.randomize(Y_truth)
    z=model.forcast_data_allstep(Y_truth)
    y_noise = np.random.normal(0.0, 0.025, xpoints)
    for j in range(len(z)):
        y_true_alltime[nsteps*i+j,:] = z[j]
        y_obs_alltime[nsteps*i+j,:] = z[j] + y_noise   
np.save("y_true_alltime.npy", y_true_alltime)
np.save("y_obs_alltime.npy", y_obs_alltime)