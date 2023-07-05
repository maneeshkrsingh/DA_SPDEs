from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
from nudging.models.stochastic_Camassa_Holm import Camsholm

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
nsteps = 5
xpoints = 40
model = Camsholm(100, nsteps, xpoints)
model.setup()
X_truth = model.allocate()
_, u0 = X_truth[0].split()

Y_truth = model.allocate()
_, u0 = Y_truth[0].split()


x, = SpatialCoordinate(model.mesh)
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

N_obs = 20

y_true = model.obs().dat.data[:]
#y_fulltrue = model.obs().dat.data[:]
y_obs_full = np.zeros((N_obs, np.size(y_true)))
y_true_full = np.zeros((N_obs, np.size(y_true)))

#y_alltime = model.obs().dat.data[:]
y_true_all = np.zeros((nsteps, np.size(y_true)))
y_obs_all = np.zeros((nsteps, np.size(y_true)))
y_true_alltime = np.zeros((N_obs*nsteps, np.size(y_true)))
y_obs_alltime = np.zeros((N_obs*nsteps, np.size(y_true)))




print(y_obs_alltime.shape)



for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step
    y_true = model.obs().dat.data[:]

    y_true_full[i,:] = y_true
    y_true_data = np.save("y_true.npy", y_true_full)

    y_noise = np.random.normal(0.0, 0.05, xpoints)  

    y_obs = y_true + y_noise
    
    y_obs_full[i,:] = y_obs 

# need to save into rank 0
y_obsdata = np.save("y_obs.npy", y_obs_full)




for i in range(N_obs):
    for step in range(nsteps):
        model.randomize(Y_truth)
        model.run(Y_truth, Y_truth) # run method for every time step
        y_alltrue = model.obs().dat.data[:]
        #print('y_true_all', y_alltrue)
        

        y_true_all[step,:] = y_alltrue
        y_true_alltime[nsteps*i+step,:] = y_true_all[step,:]

        y_true_alltime_data = np.save("y_true_alltime.npy", y_true_alltime)

        y_noise = np.random.normal(0.0, 0.05, xpoints)
        # y_obsall =  y_alltrue + y_noise
        # print('y_obs_all', y_obs_all)
        y_obs_all[step,:] = y_alltrue + y_noise
        y_obs_alltime[nsteps*i+step,:] = y_obs_all[step,:]
        y_obs_alltime_data = np.save("y_obs_alltime.npy", y_obs_alltime)


# print(y_true_alltime)
# print(y_obs_alltime)