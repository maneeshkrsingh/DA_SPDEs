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

nensemble = [5]*2

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
with fd.CheckpointFile("../../DA_KS/justks_ensemble.h5", "r", comm=comm) as afile:
    mesh = afile.load_mesh("ksmesh")
#PETSc.Sys.Print(mesh.coordinates.function_space())

# load the initial ensemble
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))
with fd.CheckpointFile("../../DA_KS/justks_ensemble.h5", "r", comm=comm) as afile:
    for i in range(nensemble[erank]):
        idx = i + offset[erank]
        #print('shape of idx', idx)
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "u", idx = idx)
        u.interpolate(u0)
        #PETSc.Sys.Print(u.function_space())

def log_likelihood(y, Y):
    ll = (y-Y)**2/2.5**2/2*fd.dx
    return ll





# Load data
y_exact_all = np.load('../../DA_KS/y_true_allpoint.npy')
y_exact = np.load('../../DA_KS/y_true.npy')
y = np.load('../../DA_KS/y_obs.npy')

PETSc.Sys.Print('shape of y', y.shape)
N_obs = y.shape[0]
# N_obs = 1

yVOM = fd.Function(model.VVOM)
#print('yvomShpe', yVOM.dat.data.shape)

# prepare shared arrays for data
# ye_all_list = []
y_e_list = []
y_sim_list = []
err_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
for m in range(y.shape[0]):
    y_e_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
    
    y_e_list.append(y_e_shared)
    # ye_all_list.append(y_e_shared)
    y_sim_list.append(y_e_shared)


N_obs = 5
ys = y.shape
ya_s = y_exact_all.shape
if fd.COMM_WORLD.rank == 0:
    print(ys[0])
    Err_particle = np.zeros((np.sum(nensemble), N_obs))
    y_e = np.zeros((np.sum(nensemble), ys[0], ys[1]))
    # y_e_allx = np.zeros((np.sum(nensemble), ya_s[0], ya_s[1]))
    y_sim = np.zeros((np.sum(nensemble), ys[0], ys[1]))
Err_final = np.zeros((N_obs))
# diagnostic
diagnostics = []
### do simulation for  enesmble
# N_sim = 100
# #PETSc.Sys.Print('simulation step')
# for k in range(N_sim):
#     PETSc.Sys.Print("Simulation Step", k)
#     for i in range(nensemble[jtfilter.ensemble_rank]):
#         model.randomize(jtfilter.new_ensemble[i])
#         model.run(jtfilter.new_ensemble[i], jtfilter.new_ensemble[i]) 
#         fwd_simdata = model.obs().dat.data[:]
#         for m in range(y.shape[1]):
#             y_sim_list[m].dlocal[i] = fwd_simdata[m]
    
    
#     for m in range(y.shape[1]):
#         y_sim_list[m].synchronise()
#         if fd.COMM_WORLD.rank == 0:
#             y_sim[:, k, m] = y_sim_list[m].data()

# # # now copy last simulated value for assimilation of particles
# for i in range(nensemble[jtfilter.ensemble_rank]):
#     jtfilter.ensemble[i][0].assign(jtfilter.new_ensemble[i][0])


# do assimiliation step
#PETSc.Sys.Print('assimilation step')
for k in range(N_obs):
    PETSc.Sys.Print("Assimlation Step", k)
    yVOM.dat.data[:] = y[k,:]

    # # # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood,  ess_tol=-880.8,  diagnostics=diagnostics)

    # load truth for error comparison
    for i in range(nensemble[jtfilter.ensemble_rank]):
        Vdg = fd.FunctionSpace(mesh, "DG", 1)
        u_particle = fd.Function(Vdg)
        u_particle.interpolate(jtfilter.ensemble[i][0])
        #PETSc.Sys.Print(mesh.coordinates.function_space())
        with fd.CheckpointFile("../../DA_KS/ks_truth.h5", "r", comm=comm) as afile:
            u_true = afile.load_function(mesh, "truth", idx = k)
        #print('error', fd.norm((u_true - u_particle)))
        err_shared.dlocal[i] = (1/sum(nensemble))*fd.norm((u_true-u_particle))/fd.norm((u_true))
    err_shared.synchronise()
    if fd.COMM_WORLD.rank == 0:
        Err_particle[:, k] += err_shared.data()
        for i in range(Err_particle.shape[0]):
            Err_final[k] += Err_particle[i,k]



#     for i in range(nensemble[jtfilter.ensemble_rank]):
#         model.un.assign(jtfilter.ensemble[i][0])
#         obsdata = model.obs().dat.data[:]
#         obsdata_allx = model.un.dat.data[:]
#         for m in range(y.shape[1]):
#             y_e_list[m].dlocal[i] = obsdata[m]
#         # for m in range(y_exact_all.shape[1]):
#         #     ye_all_list[m].dlocal[i] = obsdata_allx[m]

#     for m in range(y.shape[1]):
#         y_e_list[m].synchronise()
#         if fd.COMM_WORLD.rank == 0:
#             y_e[:, k, m] = y_e_list[m].data()
    
#     # for m in range(y_exact_all.shape[1]):
#     #     ye_all_list[m].synchronise()
#     #     if fd.COMM_WORLD.rank == 0:
#     #         y_e_allx[:, k, m] = ye_all_list[m].data()

# if fd.COMM_WORLD.rank == 0:
#     #np.save("../../DA_KS/simulated_ensemble.npy", y_sim)
#     np.save("../../DA_KS/bs_ensemble.npy", y_e)
#     # np.save("../../DA_KS/All_bs_init_assimilated_ensemble.npy",y_e_allx)

if fd.COMM_WORLD.rank == 0:
    print(Err_final.shape, Err_final)

# # # # # do checkpointing for nudging afterwards
# CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
# uout = fd.Function(CG2, name="u")

# erank = jtfilter.ensemble_rank
# filename = f"../../DA_KS/checkpoint_files/ensemble_bs_justc{erank}.h5"
# with fd.CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
#     for i in range(nensemble[erank]):
#         u = jtfilter.ensemble[i][0]
#         uout.interpolate(u)
#         afile.save_function(uout, idx=i)





# with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=jtfilter.subcommunicators.comm) as afile:
#     mesh = afile.load_mesh("ksmesh")

# comm = jtfilter.subcommunicators.comm
# ecomm = jtfilter.subcommunicators.ensemble_comm
# # load the initial ensemble
# erank = jtfilter.ensemble_rank
# filename = f"../../DA_KS/checkpoint_files/ensemble_bs_justc{erank}.h5"
# with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
#     for i in range(nensemble[erank]):
#         u = jtfilter.new_ensemble[i][0]
#         upart = afile.load_function(mesh, "u", idx = i) # read locally
#         u.interpolate(upart)