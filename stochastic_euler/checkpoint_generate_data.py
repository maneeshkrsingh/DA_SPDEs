from ctypes import sizeof
from fileinput import filename
from firedrake import *
import firedrake as fd
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
N_obs = 2
n = 64
nsteps = 5
dt = 0.1
model = Euler_SD(n, nsteps=nsteps, dt = dt, noise_scale=0.25)
model.setup()
mesh = model.mesh
x = SpatialCoordinate(mesh)



############################# initilisation ############
X0_truth = model.allocate()

X0_truth[0].interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])) 


# run model for 100 times and store inital vorticity for generating data
N_time = 100
for i in range(N_time):
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)

X_truth = model.allocate()
X_truth[0].assign(X0_truth[0])
##########################################################################

# model.q1.rename("Potential vorticity")
# model.psi0.rename("Stream function")
# Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
# v = Function(Vu, name="gradperp(stream function)")

# #v = Function(Vu)

# truth = File("../../DA_Results/2DEuler/paraview_next/truth.pvd")
# truth.write(model.q1, model.psi0, v)
# u_energy = []

# Exact numerical approximation 
for i in range(N_obs):
    #print('step', i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    # model.q1.assign(X_truth[0])
    # model.psi_solver.solve()  # solved at t+1 for psi
    # v.project(model.gradperp(model.psi0))
   
    # u_energy.append(norm(v))
    # truth.write(model.q1, model.psi0, v)



# # 
# #for checkpointing
X0_pertb = model.allocate()


# small pertirbation init
a = model.rg.normal(model.R, 0., 0.1) 
b = model.rg.normal(model.R, 0., 0.1)
c = model.rg.normal(model.R, 0., 0.1) 
d = model.rg.normal(model.R, 0., 0.1)
X0_pertb[0].interpolate(a*sin(8*pi*(x[0]+c))*sin(8*pi*(x[1]-d))+0.4*b*cos(6*pi*(x[0]+c))*cos(6*pi*(x[1]-d)))

# run model for 100 times and store inital vorticity for generating data
N_time = 100
for i in range(N_time):
    model.randomize(X0_pertb)
    model.run(X0_pertb, X0_pertb)

X_new = model.allocate()
X_new[0].assign(X0_pertb[0])

# #now do checkpointg
N_chpt = 10

f_chp = Function(model.Vdg, name="f_chp")

with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'w') as afile:

    for p in range(len(nensemble)):
        for i in range(N_chpt):
            model.randomize(X_new)
            model.run(X_new, X_new)
        afile.save_function(f_chp, idx=p) # checkpoint at only ranks 