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



# load the initial ensemble
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=comm) as afile:
    for i in range(nensemble[erank]):
        idx = i + offset[erank]
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "u", idx = idx)
        u.interpolate(u0)


# filename = f"../../DA_KS/checkpoint_files/ensemble_nudge_500{erank}.h5"
# with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
#      for i in range(nensemble[erank]):
#          u = jtfilter.ensemble[i][0]
#          u0 = afile.load_function(mesh, "u", idx = i) # read locally
#          u.interpolate(u0)


#for smoother output
Q = fd.FunctionSpace(mesh, 'CG', 1)
u_out = fd.Function(Q)



def log_likelihood(y, Y):
    ll = (y-Y)**2/2.5**2/2*fd.dx
    return ll


# Load data
y_exact_all = np.load('../../DA_KS/y_true_allpoint.npy')
y_exact = np.load('../../DA_KS/y_true.npy')
y = np.load('../../DA_KS/y_obs.npy')

PETSc.Sys.Print('shape of y', y.shape)
N_obs = y.shape[0]
N_obs = 5

yVOM = fd.Function(model.VVOM)


# prepare shared arrays for data
y_e_list = []
err_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
for m in range(y.shape[0]):
    y_e_shared = ndg.SharedArray(partition=nensemble,
                                 comm=ecomm)
    y_e_list.append(y_e_shared)

ys = y.shape
if fd.COMM_WORLD.rank == 0:
    Err_particle = np.zeros((np.sum(nensemble), N_obs))
    y_e = np.zeros((np.sum(nensemble), N_obs, ys[1]))
Err_final = np.zeros((N_obs))
# diagnostic



outfile = []
#nensemble stores the number of particles per rank
for i in range(nensemble[jtfilter.ensemble_rank]):
    ensemble_idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
    outfile.append(VTKFile(f"../../DA_KS/BS/{ensemble_idx}_output.pvd", comm=jtfilter.subcommunicators.comm))


# do assimiliation step
for k in range(N_obs):
    PETSc.Sys.Print("Assimlation Step", k)
    yVOM.dat.data[:] = y[k,:]

    # # actually do the data assimilation step
    jtfilter.assimilation_step(yVOM, log_likelihood,  ess_tol=-990.8,  diagnostics=[])

    # load truth for error comparison
    for i in range(nensemble[jtfilter.ensemble_rank]):
        # VTK file 
        u_out.interpolate(jtfilter.ensemble[i][0])
        outfile[i].write(u_out, time=k)
        # save all assimilated ensemble 
        ensemble_idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
        u_enseble = fd.Function(Q, name="assimilated_particles")
        u_enseble.interpolate(jtfilter.ensemble[i][0])
        PETSc.Sys.Print("Assimlation Step", k, "ensmeble norm", fd.norm(u_enseble))
        ensemble_cp_file = f"../../DA_KS/checkpoint_files/assimilated_ensemble_cp{ensemble_idx}.h5"
        with fd.CheckpointFile(ensemble_cp_file, 'w', comm=jtfilter.subcommunicators.comm) as afile:
            #afile.save_function(u, idx=k, name=str(ensemble_idx))
            PETSc.Sys.Print(f"Saving k={k} to {ensemble_cp_file}")  # Debugging line
            afile.save_function(u_enseble, idx=k)

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


# # Initialize a dictionary to store dynamic lists
# dynamic_lists = {}

# # Define the function space
# CG1 = fd.FunctionSpace(mesh, "CG", 1)

# # Loop over ensemble members
# for ensemble_member in range(90):
#     list_name = f"list_{ensemble_member}"  # Dynamic list name
#     dynamic_lists[list_name] = []  # Initialize an empty list in the dictionary
    
#     # Construct the filename
#     filename = f"../../DA_KS/checkpoint_files/assimilated_ensemble_cp{ensemble_member}.h5"
    
#     # Open the checkpoint file
#     with fd.CheckpointFile(filename, 'r') as afile:
#         for k in range(N_obs):
#             # Load the function from the checkpoint file
#             u0 = afile.load_function(mesh, "assimilated_particles", idx=k)
            
#             # Interpolate the function to the CG1 space
#             u = fd.Function(CG1)
#             u.interpolate(u0)
            
#             # Append the interpolated function to the corresponding list
#             dynamic_lists[list_name].append(u)
   


# if fd.COMM_WORLD.rank == 0:
#     print(Err_final.shape, Err_final)
#     np.save("../../DA_KS/500_error_bs_ensemble.npy", Err_final)
#     np.save("../../DA_KS/500_bs_ensemble.npy", y_e)

# # # # # do checkpointing for future assimilations 
# CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
# uout = fd.Function(CG2, name="u")

# erank = jtfilter.ensemble_rank
# filename = f"../../DA_KS/checkpoint_files/ensemble_bs_500{erank}.h5"
# with fd.CheckpointFile(filename, 'w', comm=jtfilter.subcommunicators.comm) as afile:
#     for i in range(nensemble[erank]):
#         u = jtfilter.ensemble[i][0]
#         uout.interpolate(u)
#         afile.save_function(uout, idx=i)