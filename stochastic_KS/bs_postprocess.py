import firedrake as fd
import nudging as ndg
import numpy as np
import math 
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP
from firedrake.output import VTKFile



import pickle

with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt =params["dt"]
nu = params["nu"]
dc = params["dc"]
#dc = 2.5

nensemble = [2]*30

model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True,
               dt=dt, nu=nu, dc=dc, L=L)

nudging = False
jtfilter = ndg.jittertemp_filter(n_jitt=0, delta=0.15,
                             verbose=2, MALA=False,
                             visualise_tape=False, nudging=nudging, sigma=0.01)
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

# def log_likelihood(y, Y):
#     ll = (y-Y)**2/2.5**2/2*fd.dx
#     return ll

y = np.load('../../DA_KS/y_obs.npy')

N_obs = y.shape[0]
#N_obs = 50
# prepare shared arrays for data

err_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)


ys = y.shape
if fd.COMM_WORLD.rank == 0:
    Err_ERE_particle = np.zeros((np.sum(nensemble), N_obs))
Err_RMSE = np.zeros((N_obs))
Err_RB = np.zeros((N_obs))

CG2 = fd.FunctionSpace(mesh, "CG", 2)
u_sum = fd.Function(CG2) # local sum
u_global_sum = fd.Function(CG2) # global sum
# do postprocessing steps 
for k in range(N_obs):
    erank = jtfilter.ensemble_rank
    filename = f"../../DA_KS/BS_checkpoint_files/ensembles_assimilated/ensemble_ch_{k}_{erank}.h5"
    with fd.CheckpointFile("../../DA_KS/ks_truth.h5", "r", comm=comm) as afile:
            u_true = afile.load_function(mesh, "truth", idx = k)
    u_sum.assign(0)
    with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
        for i in range(nensemble[erank]):
            u_particle = afile.load_function(mesh, "asm_particle", idx = i) # read locally
            u_sum += u_particle
            err_shared.dlocal[i] = math.sqrt((1/sum(nensemble)))*fd.norm((u_true-u_particle))/fd.norm((u_true))
    err_shared.synchronise()      
    jtfilter.subcommunicators.allreduce(u_sum, u_global_sum)
    u_global_sum /= sum(nensemble)
    if fd.COMM_WORLD.rank == 0:
        Err_ERE_particle[:, k] = err_shared.data()
        for i in range(Err_ERE_particle.shape[0]):
            Err_RMSE[k] += Err_ERE_particle[i,k]
        Err_RB[k] = fd.norm(u_true-u_global_sum, norm_type="L1")/fd.norm(u_true, norm_type="L1")


if fd.COMM_WORLD.rank == 0:
    np.save("../../DA_KS/2000_error_RB_bs.npy", Err_RB)
    np.save("../../DA_KS/2000_error_ERE_bs.npy", Err_RMSE)

