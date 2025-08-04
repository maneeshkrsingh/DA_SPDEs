import firedrake as fd
import nudging as ndg
import numpy as np
import math
import pickle
from firedrake.petsc import PETSc
from nudging.models.stochastic_KS_CIP import KS_CIP

Print = PETSc.Sys.Print

# --- Load parameters ---
with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt = params["dt"]
nu = params["nu"]
dc = params["dc"]

nensemble = [3] * 30

# --- Model and filter setup ---
model = KS_CIP(nsteps, xpoints, seed=12353, lambdas=True, dt=dt, nu=nu, dc=dc, L=L)
nudging = False
jtfilter = ndg.jittertemp_filter(
    n_jitt=0, delta=0.15, verbose=2, MALA=False,
    visualise_tape=False, nudging=nudging, sigma=0.01
)
jtfilter.setup(nensemble, model)

# --- Communicators ---
comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm

# --- Load mesh ---
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=comm) as afile:
    mesh = afile.load_mesh("ksmesh")
allx_shape = mesh.coordinates.dat.data[:].shape
Print("Mesh size", allx_shape[0])

# --- Load initial ensemble ---
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=comm) as afile:
    for i in range(nensemble[erank]):
        idx = i + offset[erank]
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "particle_init", idx=idx)
        u.interpolate(u0)

# --- Load observations ---
y = np.load('../../DA_KS/y_obs.npy')
N_obs = y.shape[0]

# --- Shared arrays for diagnostics and ensemble output ---
err_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
spread_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
y_e_list = [ndg.SharedArray(partition=nensemble, comm=ecomm) for _ in range(allx_shape[0])]

ys = y.shape
y_e = np.zeros((np.sum(nensemble), N_obs, allx_shape[0]))

# --- Error arrays (only on rank 0) ---
if fd.COMM_WORLD.rank == 0:
    Err_ERE_particle = np.zeros((np.sum(nensemble), N_obs))
    Err_Spread_particle = np.zeros((np.sum(nensemble), N_obs))
Err_RMSE = np.zeros(N_obs)
Err_RB = np.zeros(N_obs)
Err_RES = np.zeros(N_obs)

# --- Function spaces for ensemble mean computation ---
CG2 = fd.FunctionSpace(mesh, "CG", 2)
u_sum = fd.Function(CG2)
u_global_sum = fd.Function(CG2)

# --- Main postprocessing loop ---
for k in range(N_obs):
    u_particles_local = []  # Reset for each DA step
    erank = jtfilter.ensemble_rank
    filename = f"../../DA_KS/NudgeJitt_multi_stage/ensembles_assimilated/ensemble_ch_{k}_{erank}.h5"

    # Load truth for this step
    with fd.CheckpointFile("../../DA_KS/ks_truth.h5", "r", comm=comm) as afile:
        u_true = afile.load_function(mesh, "truth", idx=k)

    u_sum.assign(0)
    # Load local ensemble members and accumulate sum
    with fd.CheckpointFile(filename, 'r', comm=jtfilter.subcommunicators.comm) as afile:
        for i in range(nensemble[erank]):
            u_particle = afile.load_function(mesh, "asm_particle", idx=i)
            u_particles_local.append(u_particle)
            u_sum += u_particle
            err_shared.dlocal[i] = (1 / sum(nensemble)) * fd.norm(u_true - u_particle) / fd.norm(u_true)

    err_shared.synchronise()
    jtfilter.subcommunicators.allreduce(u_sum, u_global_sum)
    u_global_sum /= sum(nensemble)

    # Compute spread for each local particle
    for i, u_particle in enumerate(u_particles_local):
        spread_shared.dlocal[i] = fd.norm(u_particle - u_global_sum, norm_type="L2") / fd.norm(u_true, norm_type="L2")

    spread_shared.synchronise()

    # Synchronize and collect ensemble output
    for m in range(allx_shape[0]):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

    # Collect and sum errors on rank 0
    if fd.COMM_WORLD.rank == 0:
        Err_ERE_particle[:, k] = err_shared.data()
        Err_Spread_particle[:, k] = spread_shared.data()
        Err_RMSE[k] = np.sum(Err_ERE_particle[:, k])
        Err_RES[k] = np.sum(Err_Spread_particle[:, k])
        Err_RB[k] = fd.norm(u_true - u_global_sum, norm_type="L1") / fd.norm(u_true, norm_type="L1")

        Print('Step', k)

# --- Save results (only once, after loop) ---
if fd.COMM_WORLD.rank == 0:
    np.save("../../DA_KS/nudgejitt_ensemble_allpoints.npy", y_e)
    np.save("../../DA_KS/1000_error_RB_nudgejitt.npy", Err_RB)
    np.save("../../DA_KS/1000_error_ERE_nudgejitt.npy", Err_RMSE)
    np.save("../../DA_KS/1000_error_RES_nudgejitt.npy", Err_RES)