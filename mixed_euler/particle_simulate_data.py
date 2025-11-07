from firedrake import *
from nudging import *
import numpy as np
import gc
import petsc4py
from firedrake.petsc import PETSc
from nudging.models.stochastic_mix_euler import Euler_mixSD
import os
import pickle

print = PETSc.Sys.Print


with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

os.makedirs('../../DA_Results/2DEuler_mixed/', exist_ok=True)
""" read obs from saved file Do assimilation step for tempering and jittering steps 
"""
# Load data
# psi_exact = np.load('../../DA_Results/2DEuler_mixed/psi_true_data.npy')
# psi_vel = np.load('../../DA_Results/2DEuler_mixed/psi_obs_data.npy') 



salt = params["salt"]
noise_scale = params["noise_scale"]

print(noise_scale, 'noise scale')

N_obs = params["N_obs"]
dt = params["dt"]
nx = params["xpoints"]
nsteps = params["nsteps"]

print(N_obs, nx)

nensemble = [3] * 20


model = Euler_mixSD(nx, nsteps=nsteps, mesh = False, dt = dt,  noise_scale=noise_scale, salt=salt,  lambdas=True)


# nudging = True
# jtfilter = jittertemp_filter(n_jitt=0, delta=0.15,
#                              verbose=2, MALA=False,
#                              visualise_tape=False, nudging=True, sigma=0.01)

nudging = False
jtfilter = bootstrap_filter(verbose=2)

jtfilter.setup(nensemble, model)

# communicators
comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm


# load the initial ensemble
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))


# Fix the checkpoint file path
with CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5", 
                   'r', comm=jtfilter.subcommunicators.comm) as afile:
    mesh = afile.load_mesh("mesh2d_per")
    for ilocal in range(nensemble[jtfilter.ensemble_rank]):
        iglobal = jtfilter.layout.transform_index(ilocal, itype='l', rtype='g')
        psi_chp = afile.load_function(mesh, "psi_checkpoint", idx = iglobal) # checkpoint stream
        pv_chp = afile.load_function(mesh, "pv_checkpoint", idx = iglobal) # checkpoint vorticity

        q,psi = jtfilter.ensemble[ilocal][0].subfunctions
        q.interpolate(pv_chp)
        psi.interpolate(psi_chp)
        #print('ilocal', ilocal, 'iglobal', iglobal, norm(psi))

SD = 0.025
# N_obs = psi_vel.shape[0]
# VVOM Function
psi_VOM = Function(model.VVOM) 


N_sim = N_obs # simulation step without data assimilation
pv_sim_list = []
for m in range(psi_VOM.dat.data.shape[0]):        
    pv_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    pv_sim_list.append(pv_e_shared)

print('psi value shape ', psi.dat.data.shape) # (2000, 289)


if COMM_WORLD.rank == 0:
    pv_sim = np.zeros((np.sum(nensemble), N_sim, psi_VOM.dat.data.shape[0]))



for k in range(N_sim):
    PETSc.Sys.Print("Simulation Step", k)
    for i in range(nensemble[jtfilter.ensemble_rank]):
        # copy current ensemble to new_ensemble
        print('copying ensemble member', i)
        jtfilter.new_ensemble[i][0].assign(jtfilter.ensemble[i][0])
        print('randomising and running model for ensemble member', i)
        model.randomize(jtfilter.new_ensemble[i])
        model.run(jtfilter.new_ensemble[i], jtfilter.new_ensemble[i]) 
        fwd_simdata = model.obs().dat.data[:]
        for m in range(psi_VOM.dat.data.shape[0]):
            pv_sim_list[m].dlocal[i] = fwd_simdata[m]
    
    
    for m in range(psi_VOM.dat.data.shape[0]):
        pv_sim_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            pv_sim[:, k, m] = pv_sim_list[m].data()

print('q value shape', psi_VOM.dat.data.shape)
# prepare shared arrays for data
# psi_ens_list = []
# pv_ens_list = []

if COMM_WORLD.rank == 0:
    np.save("../../DA_Results/2DEuler_mixed/simulated_ensemble.npy", pv_sim)
    

quit()





























def log_likelihood(y, Y):
    ll = (y-Y)**2/SD**2/2*dx
    #ll = (y-Y)**2*dx
    #ll = (y-Y)*dx
    return ll

for m in range(psi_vel.shape[1]):        
    psi_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    psi_ens_list.append(psi_e_shared)

    
psi_shape = psi_vel.shape
if COMM_WORLD.rank == 0:
    psi_e = np.zeros((np.sum(nensemble), psi_shape[0], psi_shape[1]))

tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-2,
    "tao_grtol": 1.0e-5,
    "tao_gttol": 1.0e-5,
}

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    psi_VOM.dat.data[:] = psi_vel[k,:]
    #PETSc.Sys.Print('true', assemble(pv_VOM*dx))
    # assimilation step
    jtfilter.assimilation_step(psi_VOM, log_likelihood)
    # jtfilter.assimilation_step(psi_VOM, log_likelihood,
    #                        ess_tol=0.8,
    #                        taylor_test=False,
    #                        tao_params=tao_params)


    # # garbage cleanup --not sure if improved speed
    PETSc.garbage_cleanup(PETSc.COMM_SELF)
    petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    petsc4py.PETSc.garbage_cleanup(model.mesh.comm)
    gc.collect()

    # to store data
    for i in range(nensemble[jtfilter.ensemble_rank]):
        obsdata = model.obs().dat.data[:]
        for m in range(psi_vel.shape[1]):
            psi_e_list[m].dlocal[i] = obsdata[m]


    for m in range(psi_vel.shape[1]):
        psi_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            psi_e[:, k, m] = psi_e_list[m].data()

if COMM_WORLD.rank == 0:
    print(psi_e.shape)
    if not nudging:
        np.save("../../DA_Results/2DEuler_mixed/mcmcwt_assimilated_Vorticity_ensemble.npy", psi_e)
    if nudging:
        np.save("../../DA_Results/2DEuler_mixed/nudge_assimilated_Vorticity_ensemble.npy", psi_e)