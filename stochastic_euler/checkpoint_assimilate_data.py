from firedrake import *
from nudging import *
import numpy as np
import gc
import petsc4py
from firedrake.petsc import PETSc

from nudging.models.stochastic_euler import Euler_SD
import os

os.makedirs('../../DA_Results/2DEuler/', exist_ok=True)
""" read obs from saved file Do assimilation step for tempering and jittering steps 
"""
# Load data
u_exact = np.load('../../DA_Results/2DEuler/u_true_data_par.npy')
u_vel = np.load('../../DA_Results/2DEuler/u_obs_data_par.npy') 

nensemble = [4]*25
n = 32
nsteps = 5
dt = 0.025

model = Euler_SD(n, nsteps=nsteps, mesh = False, dt = dt,  noise_scale=0.025, salt=True,  lambdas=True)

#MALA = False
#verbose = True
nudging = False
jtfilter = jittertemp_filter(n_jitt = 5, delta = 0.15,
                              verbose=2, MALA=False,
                              nudging=nudging)
# jtfilter = bootstrap_filter(verbose=2)

jtfilter.setup(nensemble=nensemble, model=model, residual=False)

with CheckpointFile("../../DA_Results/2DEuler/checkpoint_files/ensemble_init.h5", 
                       'r', comm=jtfilter.subcommunicators.comm) as afile:
    mesh = afile.load_mesh("mesh2d_per")
    for ilocal in range(nensemble[jtfilter.ensemble_rank]):
        iglobal = jtfilter.layout.transform_index(ilocal, itype='l', rtype='g')
        f = afile.load_function(mesh, "f_chp", idx = iglobal) # checkpoint 
        q = jtfilter.ensemble[ilocal][0]
        q.interpolate(f)
        #print('ilocal', ilocal, 'iglobal', iglobal, norm(q))

SD = 0.005

def log_likelihood(y, Y):
    ll =(y-Y)**2/SD**2/2*dx
    return ll

N_obs = u_vel.shape[0]
# VVOM Function
u_VOM = Function(model.VVOM) 
# prepare shared arrays for data
u1_e_list = []
u2_e_list = []

for m in range(u_vel.shape[1]):        
    u1_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    u2_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    
    u1_e_list.append(u1_e_shared)
    u2_e_list.append(u2_e_shared)
    
ushape = u_vel.shape
if COMM_WORLD.rank == 0:
    u1_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))
    u2_e = np.zeros((np.sum(nensemble), ushape[0], ushape[1]))

#################################
### digonstic details
# results in a diagnostic
class samples(base_diagnostic):
    def compute_diagnostic(self, particle):
        model.q0.assign(particle[0])
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
    # PETSc.Sys.Print(u_VOM.dat.data.shape, u_vel.shape)
    u_VOM.dat.data[:,0] = u_vel[k,:,0]
    u_VOM.dat.data[:,1] = u_vel[k,:,1]

    # jtfilter.assimilation_step(u_VOM, log_likelihood)

    jtfilter.assimilation_step(u_VOM, log_likelihood,
                           ess_tol=0.8)
    # garbage cleanup --not sure if improved speed
    PETSc.garbage_cleanup(PETSc.COMM_SELF)
    petsc4py.PETSc.garbage_cleanup(model.mesh._comm)
    petsc4py.PETSc.garbage_cleanup(model.mesh.comm)
    gc.collect()

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.q0.assign(jtfilter.ensemble[i][0])
        obsdata1 = model.obs().dat.data[:][:,0]
        obsdata2 = model.obs().dat.data[:][:,1]
        for m in range(u_vel.shape[1]):
            u1_e_list[m].dlocal[i] = obsdata1[m]
            u2_e_list[m].dlocal[i] = obsdata2[m]

    for m in range(u_vel.shape[1]):
        u1_e_list[m].synchronise()
        u2_e_list[m].synchronise()
        if COMM_WORLD.rank == 0:
            u1_e[:, k, m] = u1_e_list[m].data()
            u2_e[:, k, m] = u2_e_list[m].data()
            
# #PETSc.Sys.Print("--- %s seconds ---" % (time.time() - start_time))
if COMM_WORLD.rank == 0:
    u_e = np.stack((u1_e,u2_e), axis = -1)
    print(u_e.shape)
    if not nudging:
        np.save("../../DA_Results/2DEuler/mcmcwt_assimilated_Velocity_ensemble.npy", u_e)
    if nudging:
        np.save("../../DA_Results/2DEuler/nudge_assimilated_Velocity_ensemble.npy", u_e)