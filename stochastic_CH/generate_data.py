from ctypes import sizeof
from fileinput import filename
from firedrake import *
from pyop2.mpi import MPI
from nudging import *
import numpy as np
import matplotlib.pyplot as plt
from nudging.models.stochastic_Camassa_Holm import Camsholm

import os
os.makedirs('../../DA_Results/non-smoothDA/', exist_ok=True)


"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get obseravation
add observation noise N(0, sigma^2)
"""
N_obs = 100
nsteps = 5
xpoints = 40
model = Camsholm(100, nsteps, xpoints, seed=12345, noise_scale = 0.5,  lambdas=False)
model.setup()
obs_shape = model.obs().dat.data[:]
#print(obs_shape)
x, = SpatialCoordinate(model.mesh)
#print(model.mesh.coordinates.dat.data[:])

###################  To generate data only for obs data points 
X_truth = model.allocate()
_, u0 = X_truth[0].split()
u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

truth = File("../../DA_Results/non-smoothDA/paraview/truth.pvd")

x_obs = np.linspace(0, 40, num=100, endpoint=False)
#print(x_obs)
x_obs_list = []
for i in x_obs:
    x_obs_list.append([i])
VOM = VertexOnlyMesh(model.mesh, x_obs_list)
VVOM = FunctionSpace(VOM, "DG", 0)
Z = Function(VVOM)
# def obs(self):
#         m, u = self.w0.split()
#         Y = fd.Function(self.VVOM)
#         Y.interpolate(u)
#         return Y

# # save data at obs points
y_true_allpoints = np.zeros((N_obs, 100))
y_true = np.zeros((N_obs, np.size(obs_shape)))
y_obs = np.zeros((N_obs, np.size(obs_shape)))

for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth) # run method for every time step
    _,z = X_truth[0].split()
    truth.write(z)
    Zall= Z.interpolate(z).dat.data[:]
    y = model.obs().dat.data[:]
    y_true[i,:] = y
    y_true_allpoints[i,:] = Zall

    y_noise = np.random.normal(0.0, 0.05, xpoints)  
    y_obs[i,:] =  y + y_noise 


# print(Zall)
# print(y)

print(y_true.shape)
np.save("../../DA_Results/non-smoothDA/y_true.npy", y_true)
np.save("../../DA_Results/non-smoothDA/y_obs.npy", y_obs)
np.save("../../DA_Results/non-smoothDA/y_true_all_x.npy", y_true_allpoints)
# ###################  To generate data for all time  data points 
# Y_truth = model.allocate()
# _, u0 = Y_truth[0].split()
# u0.interpolate(0.2*2/(exp(x-403./15.) + exp(-x+403./15.)) + 0.5*2/(exp(x-203./15.)+exp(-x+203./15.)))

# # save data at all points
# y_true_alltime = np.zeros((N_obs*nsteps, np.size(obs_shape)))
# y_obs_alltime = np.zeros((N_obs*nsteps, np.size(obs_shape)))

# for i in range(N_obs):
#     z = []
#     model.randomize(Y_truth)
#     z=model.forcast_data_allstep(Y_truth)
#     y_noise = np.random.normal(0.0, 0.025, xpoints)
#     for j in range(len(z)):
#         y_true_alltime[nsteps*i+j,:] = z[j]
#         y_obs_alltime[nsteps*i+j,:] = z[j] + y_noise
#         if j == len(z)-1:
#            y_true[i,:] =  y_true_alltime[nsteps*i+j,:]
#            y_obs[i,:] =  y_true[i,:] + y_noise



# np.save("y_true_alltime.npy", y_true_alltime)
# np.save("y_obs_alltime.npy", y_obs_alltime)