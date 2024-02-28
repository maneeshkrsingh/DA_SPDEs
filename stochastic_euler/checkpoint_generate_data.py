from ctypes import sizeof
from fileinput import filename
from firedrake import *
import firedrake as fd
from pyop2.mpi import MPI
import numpy as np
import matplotlib.pyplot as plt

from nudging.models.stochastic_euler import Euler_SD
import os

os.makedirs('../../DA_Results/2DEuler/', exist_ok=True)
os.makedirs('../../DA_Results/2DEuler/checkpoint_files/', exist_ok=True)

"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
#np.random.seed(138)
nensemble = [5]*20
N_obs = 50
n = 16
nsteps = 3
dt = 0.1
model = Euler_SD(n, nsteps=nsteps, dt = dt, noise_scale=0.25)
model.setup()

mesh = model.mesh
x = SpatialCoordinate(mesh)



############################# initilisation ############
X0_truth = model.allocate()

X0_truth[0].interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])) 


# run model for 100 times and store inital vorticity for generating data
N_time = 20
for i in range(N_time):
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)

X_truth = model.allocate()
X_truth[0].assign(X0_truth[0])
##########################################################################


v_true = model.obs().dat.data[:]

v1_true = v_true[:,0]
v2_true = v_true[:,1]

u1_true_all = np.zeros((N_obs, np.size(v1_true)))
u2_true_all = np.zeros((N_obs, np.size(v2_true)))
u1_obs_all = np.zeros((N_obs, np.size(v1_true)))
u2_obs_all = np.zeros((N_obs, np.size(v2_true)))


# Exact numerical approximation 
for i in range(N_obs):
    model.randomize(X_truth)
    model.run(X_truth, X_truth)

    u_VOM = model.obs()
   
    u = u_VOM.dat.data[:]

    u1_true_all[i,:] = u[:,0]
    u2_true_all[i,:] = u[:,1]

    u_1_noise = np.random.normal(0.0, 0.25, (n+1)**2 ) # mean = 0, sd = 0.05
    u_2_noise = np.random.normal(0.0, 0.25, (n+1)**2 ) 

    u1_obs = u[:,0] + u_1_noise
    u2_obs = u[:,1] + u_2_noise


    u1_obs_all[i,:] = u1_obs
    u2_obs_all[i,:] = u2_obs

u_true_all = np.stack((u1_true_all, u2_true_all), axis=-1)
u_obs_all = np.stack((u1_obs_all, u2_obs_all), axis=-1)

np.save("../../DA_Results/2DEuler/u_true_data_new.npy", u_true_all)
np.save("../../DA_Results/2DEuler/u_obs_data_new.npy", u_obs_all)


# #for checkpointing
X0_pertb = model.allocate()

# small pertirbation init
a = model.rg.normal(model.R, 0., 0.1) 
b = model.rg.normal(model.R, 0., 0.1)
c = model.rg.normal(model.R, 0., 0.1) 
d = model.rg.normal(model.R, 0., 0.1)
X0_pertb[0].interpolate(a*sin(8*pi*(x[0]+c))*sin(8*pi*(x[1]-d))+0.4*b*cos(6*pi*(x[0]+c))*cos(6*pi*(x[1]-d)))

# run model for 100 times and store inital vorticity for generating data
N_time = 20
for i in range(N_time):
    model.randomize(X0_pertb)
    model.run(X0_pertb, X0_pertb)

X_new = model.allocate()
X_new[0].assign(X0_pertb[0])

# #now do checkpointg

comm=MPI.COMM_WORLD
Lx = 2.0*fd.pi  # Zonal length
Ly = 2.0*fd.pi  # Meridonal length
mesh = fd.PeriodicRectangleMesh(n, n,Lx, Ly, direction="x",quadrilateral=True, comm=comm, name ="mesh2d_per")


N_e = 10
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
f_chp = Function(Vdg, name="f_chp")

offset_list = []
for i_rank in range(len(nensemble)):
    offset_list.append(sum(nensemble[:i_rank]))

with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'w') as afile:
    afile.save_mesh(mesh)  # optional
    for p in range(len(nensemble)):
        N_chpt = (N_e+p)*10
        for i in range(N_chpt):
            model.randomize(X_new)
            model.run(X_new, X_new)
        f_chp.interpolate(X_new[0])
        afile.save_function(f_chp, idx=p) # checkpoint only ranks 