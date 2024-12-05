import firedrake as fd
import nudging as ndg
import numpy as np
import matplotlib.pyplot as plt
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP

Print = PETSc.Sys.Print

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
                             visualise_tape=False, nudging=nudging, sigma=0.1)
# jtfilter = ndg.bootstrap_filter(verbose=2)



jtfilter.setup(nensemble, model)


# load mesh
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=jtfilter.subcommunicators.comm) as afile:
    mesh = afile.load_mesh("ksmesh")

comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm
# load the initial ensemble
erank = jtfilter.ensemble_rank
filename = f"../../DA_KS/checkpoint_files/ensemble_temp{erank}.h5"
with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
    for i in range(nensemble[erank]):
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "u", idx = i) # read locally
        u.interpolate(u0)

def log_likelihood(y, Y):
    ll = (y-Y)**2/2.5**2/2*fd.dx
    return ll


# Load data
y_exact = np.load('../../DA_KS/y_true.npy')
y = np.load('../../DA_KS/y_obs.npy')

Print('shape of y', y.shape)
N_obs = y.shape[0]
# # N_obs = 1
# N_obs = 250

yVOM = fd.Function(model.VVOM)
#print('yvomShpe', yVOM.dat.data.shape)

# prepare shared arrays for data
y_e_list = []
y_sim_list = []
for m in range(y.shape[0]):
    y_e_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
    y_e_list.append(y_e_shared)
    y_sim_list.append(y_e_shared)

ys = y.shape
if fd.COMM_WORLD.rank == 0:
    y_e = np.zeros((np.sum(nensemble), N_obs, ys[1]))
    y_sim = np.zeros((np.sum(nensemble), ys[0], ys[1]))

# diagnostic


class samples(ndg.base_diagnostic):
    def compute_diagnostic(self, particle):
        model.un.assign(particle[0])
        return model.obs().dat.data[0]

nolambdasamples = samples(ndg.Stage.WITHOUT_LAMBDAS,
                          jtfilter.subcommunicators,
                          nensemble)

resamplingsamples = samples(ndg.Stage.AFTER_ASSIMILATION_STEP,
                            jtfilter.subcommunicators,
                            nensemble)
nudgingsamples = samples(ndg.Stage.AFTER_NUDGING,
                         jtfilter.subcommunicators,
                         nensemble)
aftertempering = samples(ndg.Stage.AFTER_TEMPER_RESAMPLE,
                         jtfilter.subcommunicators,
                         nensemble)
afterjittering = samples(ndg.Stage.AFTER_ONE_JITTER_STEP,
                          jtfilter.subcommunicators,
                          nensemble)

#diagnostics = [aftertempering, afterjittering, resamplingsamples]

diagnostics = []
lis_tao_params = []

for step in range(nsteps):
    if step == 0:
        tao_params = {
            "tao_type": "lmvm",
            "tao_monitor": None,
            "tao_converged_reason": None,
            "tao_gatol": 1.0e-2,
            "tao_grtol": 1.0e-3,
            "tao_gttol": 1.0e-3,
            }
    elif step == 1:
        tao_params = {
            "tao_type": "lmvm",
            "tao_monitor": None,
            "tao_converged_reason": None,
            "tao_gatol": 1.0e-2,
            "tao_grtol": 1.0e-3,
            "tao_gttol": 1.0e-3,
            }
    elif step == 2:
        tao_params = {
            "tao_type": "lmvm",
            "tao_monitor": None,
            "tao_converged_reason": None,
            "tao_gatol": 1.0e-2,
            "tao_grtol": 1.0e-2,
            "tao_gttol": 1.0e-2,
            }
    else:
        tao_params = {
            "tao_type": "lmvm",
            "tao_monitor": None,
            "tao_converged_reason": None,
            "tao_gatol": 1.0e-1,
            "tao_grtol": 1.0e-2,
            "tao_gttol": 1.0e-2,
            }
    lis_tao_params.append(tao_params)


# To  count temping steps
temp_count = []


# do assimiliation step
#PETSc.Sys.Print('assimilation step')
for k in range(N_obs):
    Print("Assimlation Step", k)
    yVOM.dat.data[:] = y[k,:]


    # # # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood, diagnostics=diagnostics,
                                ess_tol=0.8,
                                taylor_test=False,
                                tao_params=lis_tao_params)
    temp_count.append(jtfilter.temper_count)
    #Print('temp_count', np.array(temp_count))

    for i in range(nensemble[jtfilter.ensemble_rank]):
        model.un.assign(jtfilter.ensemble[i][0])
        obsdata = model.obs().dat.data[:]
        for m in range(y.shape[1]):
            y_e_list[m].dlocal[i] = obsdata[m]

    for m in range(y.shape[1]):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

# if fd.COMM_WORLD.rank == 0:
#     np.save("../../DA_KS/nudgetemp_sigma01_count.npy", np.array(temp_count))


if fd.COMM_WORLD.rank == 0:
   
    print("ensemble shape", y_e.shape)
    if not nudging:
        np.save("../../DA_KS/500bsaftertemp_assimilated_ensemble.npy", y_e)
    if nudging:
        np.save("../../DA_KS/500quadnudge_sigma1_assimilated_ensemble.npy", y_e)
 # 

# # do checkpointing for nudging one more time
CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
uout = fd.Function(CG2, name="u")

erank = jtfilter.ensemble_rank
filename = f"../../DA_KS/checkpoint_files/ensemble_nudge_sigma1_500{erank}.h5"
with fd.CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
    for i in range(nensemble[erank]):
        u = jtfilter.ensemble[i][0]
        uout.interpolate(u0)
        afile.save_function(uout, idx=i)