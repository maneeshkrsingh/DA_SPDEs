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

truth_init = File("../../DA_Results/2DEuler/paraview_init_new/truth_init.pvd")
truth_init_ptb = File("../../DA_Results/2DEuler/paraview_init_new/truth_init_ptb.pvd")
particle_init = File("../../DA_Results/2DEuler/paraview_init_new/particle_init.pvd")
#np.random.seed(138)
nensemble = [6]*2
N_obs = 5
N_init = 5
n = 128
nsteps = 5
dt = 0.01
comm=fd.COMM_WORLD


mesh = fd.UnitSquareMesh(n, n, quadrilateral = True, comm=comm, name ="mesh2d_per")
model = Euler_SD(n, nsteps=nsteps, mesh = mesh, dt = dt, noise_scale=0.25)

model.setup()

mesh = model.mesh
x = SpatialCoordinate(mesh)

model.q1.rename("Potential vorticity")
model.psi0.rename("Stream function")
Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
v = Function(Vu, name="gradperp(stream function)")


############################# initilisation ############
X0_truth = model.allocate()

X0_truth[0].interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1]))


# run model for 100 (20*5)  times and store inital vorticity for generating data

for i in range(N_init):
    print('init_step', i)
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)


X_truth = model.allocate()
X_truth[0].assign(X0_truth[0])

model.q1.assign(X_truth[0])
model.psi_solver.solve()  # solved at t+1 for psi
v.project(model.gradperp(model.psi0))
truth_init.write(model.q1, model.psi0, v)
#print(norm(X_truth[0]))
##########################################################################


v_true = model.obs().dat.data[:]

v1_true = v_true[:,0]
v2_true = v_true[:,1]

u1_true_all = np.zeros((N_obs, np.size(v1_true)))
u2_true_all = np.zeros((N_obs, np.size(v2_true)))
u1_obs_all = np.zeros((N_obs, np.size(v1_true)))
u2_obs_all = np.zeros((N_obs, np.size(v2_true)))


# model.q1.rename("Potential vorticity")
# model.psi0.rename("Stream function")
# Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
# v = Function(Vu, name="gradperp(stream function)")

truth = File("../../DA_Results/2DEuler/paraview_init_new/truth.pvd")
truth.write(model.q1, model.psi0, v)
u_energy = []
# Exact numerical approximation 
for i in range(N_obs):
    print('In N_obs step', i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    model.q1.assign(X_truth[0])
    model.psi_solver.solve()  # solved at t+1 for psi
    v.project(model.gradperp(model.psi0))
    #print(norm(v))
    u_energy.append(norm(v))
    truth.write(model.q1, model.psi0, v)


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


u_Energy = np.array((u_energy))
np.save("../../DA_Results/2DEuler/u_true_data_new.npy", u_true_all)
np.save("../../DA_Results/2DEuler/u_obs_data_new.npy", u_obs_all)
np.save("../../DA_Results/2DEuler/u_energy_new.npy", u_Energy)

# To create initilaztion of particles, setup checkpointing
# small pertirbation in init
a = model.rg.normal(model.R, 0., 0.1) 
b = model.rg.normal(model.R, 0., 0.1)
# c = model.rg.normal(model.R, 0., 0.1) 
# d = model.rg.normal(model.R, 0., 0.1)

X0_pertb = model.allocate()
X0_pertb[0].interpolate((1+a)*sin(8*pi*(x[0]))*sin(8*pi*(x[1]))+0.4*(1+b)*cos(6*pi*(x[0]))*cos(6*pi*(x[1])))

# run model for 100 times and store inital vorticity for generating data
for i in range(N_init):
    print('In pertb step', i)
    model.randomize(X0_pertb)
    model.run(X0_pertb, X0_pertb)

X_new = model.allocate()
X_new[0].assign(X0_pertb[0])
model.q1.assign(X_new[0])
model.psi_solver.solve()  # solved at t+1 for psi
v.project(model.gradperp(model.psi0))
truth_init_ptb.write(model.q1, model.psi0, v)



# #now do checkpointing
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
f_chp = Function(Vdg, name="f_chp")


#dump data 
ndump = 5
p_dump = 0

with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'w') as afile:
    afile.save_mesh(mesh)
    for i in range(sum(nensemble)*ndump):
        model.randomize(X_new)
        model.run(X_new, X_new)
        if i % ndump == 0:
            print('checkpoint particle', p_dump , 'init')
            f_chp.interpolate(X_new[0])
            model.q1.assign(X_new[0])
            model.psi_solver.solve()  # solved at t+1 for psi
            v.project(model.gradperp(model.psi0))
            particle_init.write(model.q1, model.psi0, v)
            afile.save_function(f_chp, idx=p_dump) 
            p_dump += 1
