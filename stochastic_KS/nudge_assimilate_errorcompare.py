import firedrake as fd
import nudging as ndg
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP
from firedrake.output import VTKFile
import pickle

import os

#os.makedirs('../../DA_KS/Temp_checkpoint_files/ensembles_assimilated/', exist_ok=True)

#import os
if fd.COMM_WORLD.rank == 0:
    os.makedirs('../../DA_KS/Temp_multi_stage/ensembles_assimilated/', exist_ok=True)
fd.COMM_WORLD.barrier()  # Wait for directory creation

with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt =params["dt"]
nu = params["nu"]
dc = params["dc"]
#dc = 2.5

nensemble = [3]*30

model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True,
               dt=dt, nu=nu, dc=dc, L=L)

nudging = True
jtfilter = ndg.jittertemp_filter(n_jitt=5, delta=0.15,
                             verbose=3, MALA=False,
                             visualise_tape=False, nudging=nudging, sigma=0.001)
# jtfilter = ndg.bootstrap_filter(verbose=2)

jtfilter.setup(nensemble, model)

# communicators
comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm

# load mesh
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=comm) as afile:
    mesh = afile.load_mesh("ksmesh")

# load the initial ensemble
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=comm) as afile:
    for i in range(nensemble[erank]):
        idx = i + offset[erank]
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "particle_init", idx = idx)
        u.interpolate(u0)


# filename = f"../../DA_KS/Nudge_multi_stage/ensemble_nudge_150{erank}.h5"
# with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
#      for i in range(nensemble[erank]):
#          u = jtfilter.ensemble[i][0]
#          u0 = afile.load_function(mesh, "nudge150u", idx = i) # read locally
#          u.interpolate(u0)


def log_likelihood(y, Y):
    ll = (y-Y)**2/1.05**2/2*fd.dx
    return ll


# Load data
y_exact = np.load('../../DA_KS/y_true.npy')
y = np.load('../../DA_KS/y_obs.npy')


PETSc.Sys.Print('shape of y', y.shape)
N_obs = y.shape[0]
N_obs = 900

yVOM = fd.Function(model.VVOM)


# prepare shared arrays for data
y_e_list = []
err_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
for m in range(y.shape[1]):
    y_e_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
    y_e_list.append(y_e_shared)

ys = y.shape
if fd.COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), N_obs, ys[1]))

# diagnostic
diagnostics = []

outfile = []
# #nensemble stores the number of particles per rank
for i in range(nensemble[jtfilter.ensemble_rank]):
    ensemble_idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
    outfile.append(VTKFile(f"../../DA_KS/Temp_vtk/{ensemble_idx}_output.pvd", comm=jtfilter.subcommunicators.comm))


tao_params = {
    "tao_type": "lmvm",          # LMVM solver
    "tao_monitor": None,
    "tao_converged_reason": None,
    #"tao_lmvm_max": 15,           # Reduce memory usage by limiting the number of updates
    "tao_gatol": 1.0e-2,         # Gradient tolerance
    "tao_grtol": 1.0e-3,         # Relative gradient tolerance
    "tao_gttol": 1.0e-3,         # Trust-region tolerance
}


ess_list = []

# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Assimlation Step", k)
    yVOM.dat.data[:] = y[k,:]

    # # # actually do the data assimilation step
    #jtfilter.assimilation_step(yVOM, log_likelihood, diagnostics=diagnostics)


    jtfilter.assimilation_step(yVOM, log_likelihood, diagnostics=diagnostics,
                                ess_tol=0.8,
                                taylor_test=False,
                                tao_params=tao_params)
    # ess_list.append(jtfilter.ess_count)
    # np.save("../../DA_KS/ess.npy", np.array(ess_list))
    # # # # do checkpointing for future assimilations 
    CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
    uout = fd.Function(CG2, name="asm_particle")

    erank = jtfilter.ensemble_rank
    filename = f"../../DA_KS/Temp_multi_stage/ensembles_assimilated/ensemble_ch_{k}_{erank}.h5"
    with fd.CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
        for i in range(nensemble[erank]):
            u = jtfilter.ensemble[i][0]
            uout.interpolate(u)
            afile.save_function(uout, idx=i)
    # # save VTK
    for i in range(nensemble[jtfilter.ensemble_rank]):
        particle_vtk = fd.Function(model.Vdg, name='asmlted_particles')
        particle_vtk.interpolate(jtfilter.ensemble[i][0])
        outfile[i].write(particle_vtk, time=k)

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.un.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]
        
    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()
            np.save("../../DA_KS/tempjitt_ensemble.npy", y_e)




# # # # # do checkpointing for future assimilations 
# CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
# uout = fd.Function(CG2, name="nudge300u")

# erank = jtfilter.ensemble_rank
# filename = f"../../DA_KS/Nudge_multi_stage/ensemble_nudge_300{erank}.h5"
# with fd.CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
#     for i in range(nensemble[erank]):
#         u = jtfilter.ensemble[i][0]
#         uout.interpolate(u)
#         afile.save_function(uout, idx=i)
