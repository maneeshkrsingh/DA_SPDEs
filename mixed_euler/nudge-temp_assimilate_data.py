from firedrake import *
from nudging import *
import numpy as np
import gc
import petsc4py
from firedrake.petsc import PETSc

from nudging import  base_diagnostic, Stage

from nudging.models.stochastic_mix_euler import Euler_mixSD
import os

os.makedirs('../../DA_Results/2DEuler_mixed/', exist_ok=True)
""" read obs from saved file 
Do assimilation step for tempering and jittering steps 
"""
# Load data
psi_exact = np.load('../../DA_Results/2DEuler_mixed/psi_true_data.npy')
psi_vel = np.load('../../DA_Results/2DEuler_mixed/psi_obs_data.npy') 

nensemble = [2]*30
n = 32
nsteps = 5
dt = 1/20

model = Euler_mixSD(n, nsteps=nsteps, mesh = False, dt = dt,  noise_scale=1.05, salt=False,  lambdas=True)


# nudging = True
# jtfilter = jittertemp_filter(n_jitt=0, delta=0.15,
#                              verbose=2, MALA=False,
#                              visualise_tape=False, nudging=nudging, sigma=0.001)

nudging = False
jtfilter = bootstrap_filter(verbose=2)

jtfilter.setup(nensemble=nensemble, model=model, residual=False)


with CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5", 
                       'r', comm=jtfilter.subcommunicators.comm) as afile:
    mesh = afile.load_mesh("mesh2d_per")


erank = jtfilter.ensemble_rank
filename = f"../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_temp{erank}.h5"
with CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
    for ilocal in range(nensemble[erank]):
        psi_chp = afile.load_function(mesh, "psi_ensemble", idx = ilocal) # checkpoint stream
        pv_chp = afile.load_function(mesh, "pv_ensemble", idx = ilocal) # checkpoint vorticity
        q,psi = jtfilter.ensemble[ilocal][0].split()
        q.interpolate(pv_chp)
        psi.interpolate(psi_chp)
        print('Rank', erank, 'ilocal', ilocal, norm(q))

SD = 0.01

def log_likelihood(y, Y):
    ll = (y-Y)**2/SD**2/2*dx
    return ll

N_obs = psi_vel.shape[0]
# VVOM Function
psi_VOM = Function(model.VVOM) 
# prepare shared arrays for data
psi_e_list = []

for m in range(psi_vel.shape[1]):        
    psi_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    psi_e_list.append(psi_e_shared)

    
psi_shape = psi_vel.shape
if COMM_WORLD.rank == 0:
    psi_e = np.zeros((np.sum(nensemble), psi_shape[0], psi_shape[1]))

tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-2,
    "tao_grtol": 1.0e-50,
    "tao_gttol": 1.0e-5,
}

# add sample class 
class samples(base_diagnostic):
    def compute_diagnostic(self, particle):
        model.qpsi0.assign(particle[0])
        return model.obs().dat.data[0]

resamplingsamples = samples(Stage.AFTER_ASSIMILATION_STEP,
                            jtfilter.subcommunicators,
                            nensemble)
nudgingsamples = samples(Stage.AFTER_NUDGING,
                         jtfilter.subcommunicators,
                         nensemble)
nolambdasamples = samples(Stage.WITHOUT_LAMBDAS,
                          jtfilter.subcommunicators,
                          nensemble)


diagnostics = [nudgingsamples,
               resamplingsamples,
               nolambdasamples]

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    psi_VOM.dat.data[:] = psi_vel[k,:]
    #PETSc.Sys.Print('true', assemble(pv_VOM*dx))
    # assimilation step
    jtfilter.assimilation_step(psi_VOM, log_likelihood)
    # jtfilter.assimilation_step(psi_VOM, log_likelihood,
    #                            diagnostics=diagnostics,
    #                             ess_tol=-666,
    #                             taylor_test=False,
    #                             tao_params=tao_params)


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

if jtfilter.subcommunicators.global_comm.rank == 0:
    before, descriptors = nolambdasamples.get_archive()
    after, descriptors = nudgingsamples.get_archive()
    resampled, descriptors = resamplingsamples.get_archive()

    np.save("before", before)
    np.save("after", after)
    np.save("resampled", resampled)


if COMM_WORLD.rank == 0:
    print(psi_e.shape)
    if not nudging:
        np.save("../../DA_Results/2DEuler_mixed/mcmcwt_assimilated_Vorticity_ensemble.npy", psi_e)
    if nudging:
        np.save("../../DA_Results/2DEuler_mixed/nudge_assimilated_Vorticity_ensemble.npy", psi_e)