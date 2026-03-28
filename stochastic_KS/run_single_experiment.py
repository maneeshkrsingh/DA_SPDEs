"""
Run a single KS DA experiment from checkpoint at step 675, for 75 steps.
Saves ensemble checkpoints and full field data at all mesh points.
Usage: mpiexec -n 30 python run_single_experiment.py <model_seed> <resampler_seed> <exp_num> <output_dir> <filter_type>
filter_type: 'nudge' or 'tempjitt'
Run from: ~/DA_SPDEs/stochastic_KS/
Firedrake venv must be activated before running.
"""
import sys
import os
import firedrake as fd
import nudging as ndg
import numpy as np
from firedrake.petsc import PETSc
from pyop2.mpi import MPI
from firedrake.__future__ import interpolate
from nudging.models.stochastic_KS_CIP import KS_CIP
import pickle
import time

Print = PETSc.Sys.Print

# ---- Parse arguments ----
model_seed = int(sys.argv[1])
resampler_seed = int(sys.argv[2])
exp_num = int(sys.argv[3])
output_dir = sys.argv[4]
filter_type = sys.argv[5]  # 'nudge' or 'tempjitt'

# ---- Configuration ----
START_STEP = 675
N_STEPS = 50
nensemble = [3] * 30
SAVE_CHECKPOINTS = False

# ---- Load model parameters ----
with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
xpoints = params["xpoints"]
L = params["L"]
dt = params["dt"]
nu = params["nu"]
dc = params["dc"]
noise_var = params["noise_var"]  # 1.1025 = 1.05^2

# ---- Checkpoint paths ----
CKPT_DIRS = {
    'nudge': '../../DA_KS/NudgeJitt_multi_stage/ensembles_assimilated',
    'tempjitt': '../../DA_KS/TempJittering/ensembles_assimilated',
}

# ---- Create output directories early ----
exp_dir = os.path.join(output_dir, filter_type, f"exp_{exp_num:03d}")
ckpt_out_dir = os.path.join(exp_dir, "checkpoints")
if fd.COMM_WORLD.rank == 0:
    os.makedirs(exp_dir, exist_ok=True)
    if SAVE_CHECKPOINTS:
        os.makedirs(ckpt_out_dir, exist_ok=True)
fd.COMM_WORLD.barrier()

# ---- Setup model and filter ----
nudging = (filter_type == 'nudge')

model = KS_CIP(nsteps, xpoints, seed=model_seed,
               lambdas=nudging, dt=dt, nu=nu, dc=dc, L=L)

if nudging:
    jtfilter = ndg.jittertemp_filter(n_jitt=5, delta=0.05,
                                      verbose=2, MALA=False,
                                      visualise_tape=False,
                                      nudging=True, sigma=0.001)
else:
    jtfilter = ndg.jittertemp_filter(n_jitt=5, delta=0.15,
                                      verbose=2, MALA=False,
                                      visualise_tape=False,
                                      nudging=False)

jtfilter.setup(nensemble, model, resampler_seed=resampler_seed)

# communicators
comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm
erank = ecomm.rank
Np = sum(nensemble)

# ---- Load mesh ----
with fd.CheckpointFile("../../DA_KS/ks_ensemble.h5", "r", comm=comm) as afile:
    mesh = afile.load_mesh("ksmesh")

allx_shape = mesh.coordinates.dat.data[:].shape
n_mesh_pts = allx_shape[0]
Print(f"Mesh size: {n_mesh_pts}")

# ---- Load ensemble checkpoint at START_STEP ----
ckpt_dir = CKPT_DIRS[filter_type]
ckpt_file = os.path.join(ckpt_dir, f"ensemble_ch_{START_STEP}_{erank}.h5")
Print(f"Loading checkpoint: {ckpt_file}")

with fd.CheckpointFile(ckpt_file, 'r', comm=comm) as afile:
    for i in range(nensemble[erank]):
        u = jtfilter.ensemble[i][0]
        u0 = afile.load_function(mesh, "asm_particle", idx=i)
        u.interpolate(u0)

# ---- Log likelihood ----
def log_likelihood(y, Y):
    return (y - Y) ** 2 / noise_var / 2 * fd.dx

# ---- Load observation data ----
y_obs = np.load('../../DA_KS/y_obs.npy')
Print('shape of y_obs', y_obs.shape)

yVOM = fd.Function(model.VVOM)

# ---- Shared arrays for ALL mesh points ----
y_e_list = []
for m in range(n_mesh_pts):
    y_e_list.append(ndg.SharedArray(partition=nensemble, comm=ecomm))

err_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
spread_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)

if fd.COMM_WORLD.rank == 0:
    y_e = np.zeros((Np, N_STEPS, n_mesh_pts))
    Err_RMSE = np.zeros(N_STEPS)
    Err_RB = np.zeros(N_STEPS)
    Err_RES = np.zeros(N_STEPS)

# ---- TAO parameters ----
tao_input = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-2,
    "tao_grtol": 1.0e-3,
    "tao_gttol": 1.0e-3,
}

diagnostics = []

# ---- Prepare functions for checkpointing and metrics ----
# Use model.mesh so ensemble functions and u_sum share the same mesh object
CG2 = fd.FunctionSpace(model.mesh, "CG", 2)
uout = fd.Function(CG2, name="asm_particle")
u_sum = fd.Function(CG2)
u_global_sum = fd.Function(CG2)

# ---- Assimilation loop ----
t_total_start = time.perf_counter()
step_times = []
if fd.COMM_WORLD.rank == 0:
    ess_list = []
for k in range(N_STEPS):
    step = START_STEP + k
    t_step_start = time.perf_counter()
    Print(f"[Exp {exp_num}] DA Step {step} ({k+1}/{N_STEPS})")

    yVOM.dat.data[:] = y_obs[step, :]

    jtfilter.assimilation_step(yVOM, log_likelihood,
                                diagnostics=diagnostics,
                                ess_tol=0.8 if not nudging else -990.8,
                                taylor_test=False,
                                tao_params=tao_input)
    # ---- save ESS ----
    if fd.COMM_WORLD.rank == 0:
        ess_list.append(jtfilter.ess)
    # ---- Save checkpoint ----
    if SAVE_CHECKPOINTS:
        ckpt_file_out = os.path.join(ckpt_out_dir,
                                      f"ensemble_ch_{step}_{erank}.h5")
        with fd.CheckpointFile(ckpt_file_out, 'w', comm=comm) as afile:
            for i in range(nensemble[erank]):
                uout.interpolate(jtfilter.ensemble[i][0])
                afile.save_function(uout, idx=i)

    # ---- Load truth for this step ----
    # load_function requires a file-loaded mesh; copy data into model.mesh function for norm ops
    with fd.CheckpointFile("../../DA_KS/ks_truth.h5", "r", comm=comm) as afile:
        u_true_file = afile.load_function(mesh, "truth", idx=step)
    u_true = fd.Function(CG2)
    u_true.dat.data[:] = u_true_file.dat.data[:]

    # ---- Collect full field data at ALL mesh points ----
    u_sum.assign(0)
    u_particles_local = []
    for i in range(nensemble[erank]):
        u_particle = jtfilter.ensemble[i][0]
        fielddata = u_particle.dat.data[:]
        for m in range(n_mesh_pts):
            y_e_list[m].dlocal[i] = fielddata[m]
        u_sum += u_particle
        err_shared.dlocal[i] = (1.0 / Np) * fd.norm(u_true - u_particle) / fd.norm(u_true)
        u_particles_local.append(u_particle)

    err_shared.synchronise()

    # global ensemble mean
    jtfilter.subcommunicators.allreduce(u_sum, u_global_sum)
    u_global_sum /= Np

    # spread per particle
    for i, u_particle in enumerate(u_particles_local):
        spread_shared.dlocal[i] = fd.norm(u_particle - u_global_sum) / fd.norm(u_true)

    spread_shared.synchronise()

    # synchronise full field data
    for m in range(n_mesh_pts):
        y_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            y_e[:, k, m] = y_e_list[m].data()

    # ---- Compute metrics on rank 0 ----
    if fd.COMM_WORLD.rank == 0:
        Err_RMSE[k] = np.sum(err_shared.data())
        Err_RES[k] = np.sum(spread_shared.data()**2) / (Np - 1)
        Err_RB[k] = fd.norm(u_true - u_global_sum, norm_type="L1") / fd.norm(u_true, norm_type="L1")

    t_step = time.perf_counter() - t_step_start
    step_times.append(t_step)
    Print(f"  Step {step} time: {t_step:.1f}s  (avg so far: {sum(step_times)/len(step_times):.1f}s)")

    PETSc.garbage_cleanup(PETSc.COMM_SELF)

t_total = time.perf_counter() - t_total_start

# ---- Save results ----
if fd.COMM_WORLD.rank == 0:
    np.save(os.path.join(exp_dir, "ensemble_allpoints.npy"), y_e)
    np.save(os.path.join(exp_dir, "RMSE.npy"), Err_RMSE)
    np.save(os.path.join(exp_dir, "RB.npy"), Err_RB)
    np.save(os.path.join(exp_dir, "RES.npy"), Err_RES)
    np.save(os.path.join(exp_dir, "step_times.npy"), np.array(step_times))
    np.save(os.path.join(exp_dir, "ESS.npy"), np.array(ess_list))
Print(f"Experiment {exp_num} complete. Total time: {t_total/60:.1f} min  ({t_total:.0f}s). Results in {exp_dir}")
