from firedrake import *
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np

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

truth_init = VTKFile("../../DA_Results/2DEuler/paraview_saltadtnoise/truth_init.pvd")
truth = VTKFile("../../DA_Results/2DEuler/paraview_saltadtnoise/truth.pvd")
truth_init_ptb = VTKFile("../../DA_Results/2DEuler/paraview_saltadtnoise/truth_init_ptb.pvd")
particle_init = VTKFile("../../DA_Results/2DEuler/paraview_saltadtnoise/particle_init.pvd")

nensemble = [10]*10
N_obs = 50
N_init = 500
n = 4
nsteps = 5
dt = 0.025


comm=fd.COMM_WORLD
mesh = fd.UnitSquareMesh(n, n, quadrilateral = True, comm=comm, name ="mesh2d_per")

#model = Euler_SD(n, nsteps=nsteps, mesh = mesh, dt = dt, noise_scale=0.025)
model = Euler_SD(n, nsteps=nsteps, mesh = mesh, dt = dt, noise_scale=.25, salt=True,  lambdas=True)


model.setup()

mesh = model.mesh
x = SpatialCoordinate(mesh)

model.q1.rename("Potential vorticity")
model.psi0.rename("Stream function")
Vu = VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
v = Function(Vu, name="gradperp(stream function)")


############################# initilisation ############
X0_truth = model.allocate()

X0_truth[0].interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])
                        +0.3*cos(10*pi*x[0])*cos(4*pi*x[1]) +0.02*sin(2*pi*x[0])+ 0.02*sin(2*pi*x[1]))


# run model for 100 (20*5)  times and store inital vorticity for generating data
# #To store velocity values 
pv_VOM = model.obs()
pv_VOM_out = Function(model.VVOM_out)
pv_VOM_out.interpolate(pv_VOM)
pv = pv_VOM_out.dat.data_ro.copy()

pv_true = pv_VOM_out.dat.data_ro.copy()
# v1_true = v_true[:,0]
# v2_true = v_true[:,1]

# To store inilization of  velocity compoenents
pv_init = np.zeros((np.size(pv_true)))


for i in range(N_init):
    PETSc.Sys.Print('========init_step=============', i)
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)

    model.q1.assign(X0_truth[0])
    model.psi_solver.solve()  # solved at t+1 for psi
    v.project(model.gradperp(model.psi0))
    #print('Energy Init', norm(v))
    truth_init.write(model.q1, model.psi0, v)

    pv_init_VOM = model.obs()
    pv_init_VOM_out = Function(model.VVOM_out)
    pv_init_VOM_out.interpolate(pv_init_VOM)
    pvinit = pv_init_VOM_out.dat.data_ro.copy()
    PETSc.Sys.Print('Init', pvinit)
    if i == N_init-1:
        if comm.rank == 0:
            pv_init = pvinit
            np.save("../../DA_Results/2DEuler/pv_init.npy", pv_init)

X_truth = model.allocate()
X_truth[0].assign(X0_truth[0])

# model.q1.assign(X_truth[0])
# model.psi_solver.solve()  # solved at t+1 for psi
# v.project(model.gradperp(model.psi0))
# #print(type(v))
# truth_init.write(model.q1, model.psi0, v)

####################################### compute Courant number ###################



pv_true_all = np.zeros((N_obs, np.size(pv_true)))
pv_obs_all = np.zeros((N_obs, np.size(pv_true)))

u_energy = []

# Forwrad run for truth

for i in range(N_obs):
    PETSc.Sys.Print('=============In N_obs step=================', i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    model.q1.assign(X_truth[0])
    model.psi_solver.solve()  # solved at t+1 for psi
    v.project(model.gradperp(model.psi0))
    print('energy in obs', norm(v))
    u_energy.append(norm(v))
    truth.write(model.q1, model.psi0, v)

    pv_VOM = model.obs()
    pv_VOM_out = Function(model.VVOM_out)
    pv_VOM_out.interpolate(pv_VOM)
    pv = pv_VOM_out.dat.data_ro.copy()

    if comm.rank == 0:
        pv_true_all[i,:]= pv
        print(pv.max(), pv.min())
        pv_noise = np.random.normal(0.0, 0.01, (n+1)**2 ) # mean = 0, sd = 0.05
        print(pv_noise.max(), pv_noise.min())
        pv_obs = pv + pv_noise


        pv_obs_all[i,:] = pv_obs

    #print(pv_obs_all)

    np.save("../../DA_Results/2DEuler/pv_true_data.npy", pv_true_all)
    np.save("../../DA_Results/2DEuler/pv_obs_data.npy", pv_obs_all)
    # np.save("../../DA_Results/2DEuler/u_energy_new.npy", u_Energy)

# u_Energy = np.array((u_energy))
# np.save("../../DA_Results/2DEuler/u_energy_new.npy", u_Energy)

################### same initialization as truth
# # #now do checkpointing
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
f_chp = Function(Vdg, name="f_chp")
#dump data 
ndump = 2
p_dump = 0

pv_particle_init = np.zeros((sum(nensemble), np.size(pv_true)))
with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'w') as afile:
    afile.save_mesh(mesh)
    for i in range(sum(nensemble)*ndump):
        model.randomize(X_truth)
        model.run(X_truth, X_truth)
        if i % ndump == 0:
            PETSc.Sys.Print('===========checkpoint particle============', p_dump , 'init')
            f_chp.interpolate(X_truth[0])
            model.q1.assign(X_truth[0])
            model.psi_solver.solve()  # solved at t+1 for psi
            v.project(model.gradperp(model.psi0))
            particle_init.write(model.q1, model.psi0, v)
            afile.save_function(f_chp, idx=p_dump)
            pv_particle_VOM = model.obs()
            pv_particle_VOM_out = Function(model.VVOM_out)
            pv_particle_VOM_out.interpolate(pv_particle_VOM)
            pv_particle = pv_particle_VOM_out.dat.data_ro.copy()

            if comm.rank == 0:
                pv_particle_init[p_dump,:] = pv_particle
                print(pv_particle_init.max(), pv_particle_init.min())
            p_dump += 1
    np.save("../../DA_Results/2DEuler/pv_particle_init.npy", pv_particle_init)





############################## pertubed initlization###################################

# # # To create initilaztion of particles, setup checkpointing
# # # small pertirbation in init
# a = model.rg.normal(model.R, 0., 0.01) 
# b = model.rg.normal(model.R, 0., 0.01)
# # c = model.rg.normal(model.R, 0., 0.1) 
# # d = model.rg.normal(model.R, 0., 0.1)

# X0_pertb = model.allocate()
# X0_pertb[0].interpolate((1+a)*sin(8*pi*(x[0]))*sin(8*pi*(x[1]))+0.4*(1+b)*cos(6*pi*(x[0]))*cos(6*pi*(x[1])))

# # run model for 100 times and store inital vorticity for generating data
# for i in range(N_init):
#     PETSc.Sys.Print('==============In pertb step====================', i)
#     model.randomize(X0_pertb)
#     model.run(X0_pertb, X0_pertb)
#     model.q1.assign(X0_pertb[0])
#     model.psi_solver.solve()  # solved at t+1 for psi
#     v.project(model.gradperp(model.psi0))
#     truth_init_ptb.write(model.q1, model.psi0, v)

# X_new = model.allocate()
# X_new[0].assign(X0_pertb[0])




# # # #now do checkpointing
# Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
# f_chp = Function(Vdg, name="f_chp")


# #dump data 
# ndump = 5
# p_dump = 0

# pv_particle_init = np.zeros((sum(nensemble), np.size(pv_true)))
# with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
#                        'w') as afile:
#     afile.save_mesh(mesh)
#     for i in range(sum(nensemble)*ndump):
#         model.randomize(X_truth)
#         model.run(X_truth, X_truth)
#         if i % ndump == 0:
#             PETSc.Sys.Print('===========checkpoint particle============', p_dump , 'init')
#             f_chp.interpolate(X_truth[0])
#             model.q1.assign(X_truth[0])
#             model.psi_solver.solve()  # solved at t+1 for psi
#             v.project(model.gradperp(model.psi0))
#             particle_init.write(model.q1, model.psi0, v)
#             afile.save_function(f_chp, idx=p_dump)
#             pv_particle_VOM = model.obs()
#             pv_particle_VOM_out = Function(model.VVOM_out)
#             pv_particle_VOM_out.interpolate(pv_particle_VOM)
#             pv_particle = pv_particle_VOM_out.dat.data_ro.copy()

#             if comm.rank == 0:
#                 pv_particle_init[p_dump,:] = pv_particle
#                 #print(pv_particle_init)
    
#             p_dump += 1