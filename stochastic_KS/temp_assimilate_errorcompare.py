import firedrake as fd
import nudging as ndg
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP

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

nudging = True
jtfilter = ndg.jittertemp_filter(n_jitt=5, delta=0.15,
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


def log_likelihood(y, Y):
    ll = (y-Y)**2/2.5**2/2*fd.dx
    return ll


# Load data
y_exact_all = np.load('../../DA_KS/y_true_allpoint.npy')
y_exact = np.load('../../DA_KS/y_true.npy')
y = np.load('../../DA_KS/y_obs.npy')

PETSc.Sys.Print('shape of y', y.shape)
N_obs = y.shape[0]
N_obs = 500

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
    Err_particle = np.zeros((np.sum(nensemble), N_obs))
    y_e = np.zeros((np.sum(nensemble), N_obs, ys[1]))
Err_final = np.zeros((N_obs))
# diagnostic
diagnostics = []




# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Assimlation Step", k)
    yVOM.dat.data[:] = y[k,:]

    # # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood,  ess_tol=0.8,  diagnostics=diagnostics)


    # load truth for error comparison
    for i in range(nensemble[jtfilter.ensemble_rank]):
        Vdg = fd.FunctionSpace(mesh, "DG", 1)
        u_particle = fd.Function(Vdg)
        u_particle.interpolate(jtfilter.ensemble[i][0])
        with fd.CheckpointFile("../../DA_KS/ks_truth.h5", "r", comm=comm) as afile:
            u_true = afile.load_function(mesh, "truth", idx = k)
        err_shared.dlocal[i] = (1/sum(nensemble))*fd.norm((u_true-u_particle))/fd.norm((u_true))
    err_shared.synchronise()
    if fd.COMM_WORLD.rank == 0:
        Err_particle[:, k] += err_shared.data()
        for i in range(Err_particle.shape[0]):
            Err_final[k] += Err_particle[i,k]

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.un.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]
        
    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()
    
if fd.COMM_WORLD.rank == 0:
    print(Err_final.shape, Err_final)
    np.save("../../DA_KS/2000_error_tempjitt_ensemble.npy", Err_final)
    np.save("../../DA_KS/2000_tempjitts_ensemble.npy", y_e)

# # # # do checkpointing for future assimilations 
CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
uout = fd.Function(CG2, name="u")

erank = jtfilter.ensemble_rank
filename = f"../../DA_KS/checkpoint_files/ensemble_tempjitt_2000{erank}.h5"
with fd.CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
    for i in range(nensemble[erank]):
        u = jtfilter.ensemble[i][0]
        uout.interpolate(u)
        afile.save_function(uout, idx=i)