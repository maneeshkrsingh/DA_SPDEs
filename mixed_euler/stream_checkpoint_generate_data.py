from firedrake import *
import firedrake as fd
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
import numpy as np
from nudging.models.stochastic_mix_euler import Euler_mixSD
import os
os.makedirs('../../DA_Results/2DEuler_mixed/', exist_ok=True)
os.makedirs('../../DA_Results/2DEuler_mixed/checkpoint_files/', exist_ok=True)
"""
create some synthetic data/observation data at T_1 ---- T_Nobs
Pick initial conditon
run model, get true value and obseravation and use paraview for viewing
add observation noise N(0, sigma^2) 
"""
truth_init = VTKFile("../../DA_Results/2DEuler_mixed/paraview_saltadtnoise/truth_init.pvd")
truth = VTKFile("../../DA_Results/2DEuler_mixed/paraview_saltadtnoise/truth.pvd")
truth_init_ptb = VTKFile("../../DA_Results/2DEuler_mixed/paraview_saltadtnoise/truth_init_ptb.pvd")
particle_init = VTKFile("../../DA_Results/2DEuler_mixed/paraview_saltadtnoise/particle_init.pvd")

nensemble = [10]*10
N_obs = 150
N_init = 250
n = 16
nsteps = 5
dt = 1/20


comm=fd.COMM_WORLD
mesh = fd.UnitSquareMesh(n, n, quadrilateral = True, comm=comm, name ="mesh2d_per")

model = Euler_mixSD(n, nsteps=nsteps, mesh = mesh, dt = dt, noise_scale=0.25, salt=False,  lambdas=True)

model.setup()
mesh = model.mesh
x = SpatialCoordinate(mesh)
############################# initilisation ############
X0_truth = model.allocate()
q0,psi0 = X0_truth[0].subfunctions
q0.interpolate(sin(8*pi*x[0])*sin(8*pi*x[1])+0.4*cos(6*pi*x[0])*cos(6*pi*x[1])
                +0.3*cos(10*pi*x[0])*cos(4*pi*x[1]) +0.02*sin(2*pi*x[0])+ 0.02*sin(2*pi*x[1]))

def gradperp(u):
    return fd.as_vector((-u.dx(1), u.dx(0)))
# #To store vorticity values 
psi_VOM = model.obs()
psi_VOM_out = Function(model.VVOM_out)
psi_VOM_out.interpolate(psi_VOM)
psi_true = psi_VOM_out.dat.data_ro.copy()
# To store inilization of  vorticity compoenents
psi_init = np.zeros((np.size(psi_true)))

### To forward run for creating initlization  for truth and particles
for i in range(N_init):
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)
    psi_VOM = model.obs()
    if i % 20 == 0:
        PETSc.Sys.Print('========Step=============', i)
        PETSc.Sys.Print('vorticity norm', fd.norm(model.q1), 'psi norm', fd.norm(model.psi1), 'noise', fd.norm(model.dU_3))
    q,psi = model.qpsi1.subfunctions
    dU = model.dU_3
    q.rename("Vorticity")
    psi.rename("stream function")
    dU.rename("noise_var")
    truth_init.write(q, psi, dU)
    # to store init data
    psi_init_VOM = model.obs()
    psi_init_VOM_out = Function(model.VVOM_out)
    psi_init_VOM_out.interpolate(psi_init_VOM)
    psinit = psi_init_VOM_out.dat.data_ro.copy()
    if i == N_init-1:
        if comm.rank == 0:
            ps_init = psinit
            np.save("../../DA_Results/2DEuler_mixed/psi_init.npy", ps_init)

### To store initlization  for truth 

X_truth = model.allocate()
X_truth[0].assign(X0_truth[0])

psi_true_all = np.zeros((N_obs, np.size(psi_true)))
psi_obs_all = np.zeros((N_obs, np.size(psi_true)))

# Forwrad run for truth
for i in range(N_obs):
    PETSc.Sys.Print('=============In N_obs step=================', i)
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    PETSc.Sys.Print('vorticity norm', fd.norm(model.q1), 'psi norm', fd.norm(model.psi1))
    q,psi = model.qpsi1.subfunctions
    q.rename("Vorticity")
    psi.rename("stream function")
    truth.write(q, psi)

    psi_VOM = model.obs()
    psi_VOM_out = Function(model.VVOM_out)
    psi_VOM_out.interpolate(psi_VOM)
    psi_true = psi_VOM_out.dat.data_ro.copy()
    
    if comm.rank == 0:
        PETSc.Sys.Print(psi_true)
        psi_true_all[i,:]= psi_true
        PETSc.Sys.Print('psi_true', psi_true.max(), psi_true.min())
        psi_noise = np.random.normal(0.0, 0.0025, (n+1)**2 ) # mean = 0, sd = 0.05
        PETSc.Sys.Print('Noise', psi_noise.max(), psi_noise.min())
        psi_obs = psi_true + psi_noise
        psi_obs_all[i,:] = psi_obs
    np.save("../../DA_Results/2DEuler_mixed/psi_true_data.npy", psi_true_all)
    np.save("../../DA_Results/2DEuler_mixed/psi_obs_data.npy", psi_obs_all)


################### same initialization as truth for all particles
X_particle = model.allocate()
X_particle[0].assign(X0_truth[0])
# # #now do checkpointing
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # PV space
f_chp = Function(Vdg, name="f_chp")
#dump data 
ndump = 5
p_dump = 0

psi_particle_init = np.zeros((sum(nensemble), np.size(psi_true)))
with fd.CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'w') as afile:
    afile.save_mesh(mesh)
    for i in range(sum(nensemble)*ndump):
        model.randomize(X_particle)
        model.run(X_particle, X_particle)
        q,psi = model.qpsi1.subfunctions
        q.rename("Vorticity")
        psi.rename("stream function")
        if i % ndump == 0:
            PETSc.Sys.Print('===========checkpoint particle============', p_dump , 'init')
            PETSc.Sys.Print('vorticity norm', fd.norm(model.q1), 'psi norm', fd.norm(model.psi1))
            f_chp.interpolate(psi)

            particle_init.write(q, psi)
            afile.save_function(f_chp, idx=p_dump)
            PETSc.Sys.Print(fd.norm(f_chp))
            psi_particle_VOM = model.obs()
            psi_particle_VOM_out = Function(model.VVOM_out)
            psi_particle_VOM_out.interpolate(psi_particle_VOM)
            psi_particle = psi_particle_VOM_out.dat.data_ro.copy()# #To store streamfunc values 


            if comm.rank == 0:
                psi_particle_init[p_dump,:] = psi_particle
                print(psi_particle_init.max(), psi_particle_init.min())
            p_dump += 1
    np.save("../../DA_Results/2DEuler_mixed/psi_particle_init.npy", psi_particle_init)