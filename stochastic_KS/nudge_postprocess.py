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

Print = PETSc.Sys.Print

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

allx_shape = mesh.coordinates.dat.data[:].shape
Print("Mesh size",allx_shape[0])
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
N_obs = 900
# prepare shared arrays for data

err_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
# prepare shared arrays for data
y_e_list = []
y_sim_list = []
err_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
for m in range(allx_shape[0]):
    y_e_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
    y_e_list.append(y_e_shared)

y_sim_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
y_sim_list.append(y_sim_shared)

ys = y.shape
y_e = np.zeros((np.sum(nensemble), N_obs, allx_shape[0]))
#Print("shape of y_e", y_e.shape)
y_sim_nstep = np.zeros((np.sum(nensemble), nsteps))
y_sim_N_obs_nstep = np.zeros((np.sum(nensemble), nsteps*N_obs))


if fd.COMM_WORLD.rank == 0:
    Err_ERE_particle = np.zeros((np.sum(nensemble), N_obs))
Err_RMSE = np.zeros((N_obs))
Err_RB = np.zeros((N_obs))

CG2 = fd.FunctionSpace(mesh, "CG", 2)
u_sum = fd.Function(CG2) # local sum
u_global_sum = fd.Function(CG2) # global sum
# do postprocessing steps 
u_true_N_obs = []
for k in range(N_obs):
    erank = jtfilter.ensemble_rank
    filename = f"../../DA_KS/Temp_multi_stage/ensembles_assimilated/ensemble_ch_{k}_{erank}.h5"
    with fd.CheckpointFile("../../DA_KS/ks_truth.h5", "r", comm=comm) as afile:
        u_true = afile.load_function(mesh, "truth", idx = k)
    u_true_N_obs.append(u_true.dat.data[:])


    # if fd.COMM_WORLD.rank == 0:
    #     print(u_true.dat.data[:].shape)
    x_obs = 2.0
    x_obs_list = [[x_obs]]
    VOM = fd.VertexOnlyMesh(mesh, x_obs_list)
    VVOM = fd.FunctionSpace(VOM, "DG", 0)
    Y = fd.Function(VVOM)
    #Y.interpolate(self.un)

    u_sum.assign(0)
    with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
        for i in range(nensemble[erank]):
            u_particle = afile.load_function(mesh, "asm_particle", idx = i) # read locally
            # to produce array
            obsdata = u_particle.dat.data[:]
            for m in range(allx_shape[0]):
                y_e_list[m].dlocal[i] = obsdata[m]
            X0 = model.allocate()
            X0[0].interpolate(u_particle)
            X1 = [fd.Function(model.V)]
            traj = model.run(X0, X1, extract_trajectory=True)
            for step in range(nsteps):
                value_at_x_obs = traj[step].at(x_obs)
                y_sim_list[0].dlocal[i] = value_at_x_obs

                y_sim_list[0].synchronise()
                #Print('Value at x_obs:', value_at_x_obs)
                if fd.COMM_WORLD.rank == 0:
                    y_sim_nstep[:, step] = y_sim_list[0].data()
                    y_sim_N_obs_nstep[:, nsteps*k+step] = \
                        y_sim_nstep[:, step]
            
            #Print("shape of obsdata", obsdata.shape)
            

            # error calculation
            u_sum += u_particle
            err_shared.dlocal[i] = math.sqrt((1/sum(nensemble)))*fd.norm((u_true-u_particle))/fd.norm((u_true))
    err_shared.synchronise()      
    jtfilter.subcommunicators.allreduce(u_sum, u_global_sum)
    u_global_sum /= sum(nensemble)


    for m in range(allx_shape[0]):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()


    # if fd.COMM_WORLD.rank == 0:
    #     Err_ERE_particle[:, k] = err_shared.data()
    #     for i in range(Err_ERE_particle.shape[0]):
    #         Err_RMSE[k] += Err_ERE_particle[i,k]
    #     Err_RB[k] = fd.norm(u_true-u_global_sum, norm_type="L1")/fd.norm(u_true, norm_type="L1")


    if fd.COMM_WORLD.rank == 0:
        Print('Step', k)
        #np.save("../../DA_KS/tester_tempjitt_ensemble_allpoints.npy", y_e)
        np.save("../../DA_KS/tester_tempjitt_simualated_all_time_obs.npy", y_sim_N_obs_nstep)
        #np.save("../../DA_KS/true_allpoints.npy",np.array(u_true_N_obs))
        # np.save("../../DA_KS/100_error_RB_nudge.npy", Err_RB)
        # np.save("../../DA_KS/100_error_ERE_nudge.npy", Err_RMSE)

