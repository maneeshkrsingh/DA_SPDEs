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
pv_exact = np.load('../../DA_Results/2DEuler_mixed/psi_true_data.npy')
pv_vel = np.load('../../DA_Results/2DEuler_mixed/psi_obs_data.npy') 



salt = params["salt"]
noise_scale = params["noise_scale"]

N_obs = params["N_obs"]
dt = params["dt"]
nx = params["xpoints"]
nsteps = params["nsteps"]

print(N_obs, nx)
nensemble = [3] * 20


model = Euler_mixSD(nx, nsteps=nsteps, mesh = False, dt = dt,  noise_scale=noise_scale, salt=salt,  lambdas=True)



nudging = True
jtfilter = jittertemp_filter(n_jitt=5, delta=0.15,
                                  verbose=3, MALA=False,
                                  visualise_tape=False, nudging=nudging, sigma=0.001)

# nudging = False
# jtfilter = bootstrap_filter(verbose=2)

jtfilter.setup(nensemble=nensemble, model=model, residual=False)

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



noise_sd = params["noise_var"] 


def log_likelihood(y, Y):
    ll = (y-Y)**2/noise_sd**2/2*dx
    #ll = (y-Y)**2*dx
    #ll = (y-Y)*dx
    return ll

#N_obs = psi_vel.shape[0]

#N_obs = 100
# VVOM Function
pv_VOM = Function(model.VVOM) 
# prepare shared arrays for data
pv_asm_list = []

for m in range(pv_vel.shape[1]):        
    pv_asm_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    pv_asm_list.append(pv_asm_shared)



    
pv_shape = pv_vel.shape

print('pvshape', pv_shape)
if COMM_WORLD.rank == 0:
    pv_e = np.zeros((np.sum(nensemble), pv_shape[0], pv_shape[1]))


# quit()
tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-2,
    "tao_grtol": 1.0e-5,
    "tao_gttol": 1.0e-5,
}

# do assimiliation step
for k in range(100):
    PETSc.Sys.Print("Step", k)
    pv_VOM.dat.data[:] = pv_vel[k,:]
    #PETSc.Sys.Print('true', assemble(pv_VOM*dx))
    # assimilation step
    # jtfilter.assimilation_step(pv_VOM, log_likelihood,diagnostics = [])
    jtfilter.assimilation_step(pv_VOM, log_likelihood,
                           ess_tol=0.8,
                           taylor_test=False,
                           tao_params=tao_params)


    # # garbage cleanup --not sure if improved speed
    PETSc.garbage_cleanup(PETSc.COMM_SELF)
    petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    petsc4py.PETSc.garbage_cleanup(model.mesh.comm)
    gc.collect()

    # to store data
    for i in range(nensemble[jtfilter.ensemble_rank]):
        obsdata = model.obs().dat.data[:]
        for m in range(pv_vel.shape[1]):
            pv_asm_list[m].dlocal[i] = obsdata[m]

    print('done storing data for step', k)
    for m in range(pv_vel.shape[1]):
        pv_asm_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            pv_e[:, k, m] = pv_asm_list[m].data()

if COMM_WORLD.rank == 0:
    print(pv_e.shape)
    if not nudging:
        np.save("../../DA_Results/2DEuler_mixed/mcmcwt_assimilated_Vorticity_ensemble.npy", pv_e)
    if nudging:
        np.save("../../DA_Results/2DEuler_mixed/nudge_assimilated_Vorticity_ensemble.npy", pv_e)