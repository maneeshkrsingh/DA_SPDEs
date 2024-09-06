from firedrake import *
from nudging import *
import numpy as np
import gc
import petsc4py
from firedrake.petsc import PETSc

from nudging.models.stochastic_mix_euler import Euler_mixSD
import os

os.makedirs('../../DA_Results/2DEuler_mixed/', exist_ok=True)
os.makedirs('../../DA_Results/2DEuler_mixed/checkpoint_files/', exist_ok=True)


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

model = Euler_mixSD(n, nsteps=nsteps,  dt = dt,  noise_scale=1.05, salt=False,  lambdas=True)


nudging = False
jtfilter = jittertemp_filter(n_jitt=4, delta=0.15,
                             verbose=2, MALA=False,
                             visualise_tape=False, nudging=nudging, sigma=0.001)

# nudging = False
# jtfilter = bootstrap_filter(verbose=2)

jtfilter.setup(nensemble=nensemble, model=model, residual=False)

with CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5", 
                       'r', comm=jtfilter.subcommunicators.comm) as afile:
    mesh = afile.load_mesh("mesh2d_per")



with CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5", 
                       'r', comm=jtfilter.subcommunicators.comm) as afile:
    #mesh = afile.load_mesh("mesh2d_per")
    for ilocal in range(nensemble[jtfilter.ensemble_rank]):
        iglobal = jtfilter.layout.transform_index(ilocal, itype='l', rtype='g')

        psi_chp = afile.load_function(mesh, "psi_chp", idx = iglobal) # checkpoint stream
        pv_chp = afile.load_function(mesh, "pv_chp", idx = iglobal) # checkpoint vorticity

        q,psi = jtfilter.ensemble[ilocal][0].split()
        #print(norm(q))
        q.interpolate(pv_chp)
        psi.interpolate(psi_chp)

SD = 0.01



def log_likelihood(y, Y):
    ll = (y-Y)**2/SD**2/2*dx
    return ll

N_obs = 10
# VVOM Function
psi_VOM = Function(model.VVOM) 

psi_e_list = []

for m in range(psi_vel.shape[1]):        
    psi_e_shared = SharedArray(partition=nensemble, 
                                 comm=jtfilter.subcommunicators.ensemble_comm)
    psi_e_list.append(psi_e_shared)

    
psi_shape = psi_vel.shape
if COMM_WORLD.rank == 0:
    psi_e = np.zeros((np.sum(nensemble), psi_shape[0], psi_shape[1]))


# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Step", k)
    psi_VOM.dat.data[:] = psi_vel[k,:]
    # assimilation step
    jtfilter.assimilation_step(psi_VOM, log_likelihood, ess_tol=0.8)

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



# # to store ensembles for further nudging
Vcg = FunctionSpace(model.mesh, "CG", 1)  # Streamfunctions
Vdg = FunctionSpace(model.mesh, "DQ", 1)  # PV space
psi_e = Function(Vcg, name="psi_ensemble") # checkpoint streamfunc
q_e = Function(Vdg, name="pv_ensemble")   # checkpoint vorticity

erank = jtfilter.ensemble_rank
filename = f"../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_temp{erank}.h5"
with CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
    for ilocal in range(nensemble[erank]):
        q,psi = jtfilter.ensemble[ilocal][0].split()
        psi_e.interpolate(psi)
        afile.save_function(psi_e, idx=ilocal)
        q_e.interpolate(q)
        afile.save_function(q_e, idx=ilocal)
        print('Rank', erank, 'ilocal', ilocal,  norm(q_e))