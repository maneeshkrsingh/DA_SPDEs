import firedrake as fd
import nudging as ndg
import numpy as np
import pickle
import os
from firedrake.petsc import PETSc
from firedrake.output import VTKFile

from nudging import base_diagnostic, Stage
from nudging.models.stochastic_mix_euler import Euler_mixSD

Print = PETSc.Sys.Print

# Make results directories
if fd.COMM_WORLD.rank == 0:
    os.makedirs('../../DA_Results/2DEuler_mixed/', exist_ok=True)
    os.makedirs('../../DA_Results/2DEuler_mixed/checkpoint_files/', exist_ok=True)
    os.makedirs('../../DA_Results/2DEuler_mixed/vtk_output/', exist_ok=True)
fd.COMM_WORLD.barrier()

# Load data
psi_exact = np.load('../../DA_Results/2DEuler_mixed/psi_true_data.npy')
psi_vel = np.load('../../DA_Results/2DEuler_mixed/psi_obs_data.npy')

with open("params.pickle", 'rb') as handle:
    params = pickle.load(handle)

nsteps = params["nsteps"]
dt = params["dt"]
n_x = params["xpoints"]
dw_scale = params["noise_scale"]
salt = params["salt"]
nensemble = params["nensemble"]

# Model
model = Euler_mixSD(n_x, nsteps=nsteps, dt=dt,
                    noise_scale=dw_scale, salt=salt, lambdas=True)

# Filter
nudging = True
jtfilter = ndg.jittertemp_filter(n_jitt=5, delta=0.15,
                                 verbose=2, MALA=False,
                                 visualise_tape=False, nudging=nudging, sigma=0.001)

jtfilter.setup(nensemble=nensemble, model=model, residual=False)

# communicators
comm = jtfilter.subcommunicators.comm
ecomm = jtfilter.subcommunicators.ensemble_comm

# load mesh
with fd.CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5",
                       'r', comm=comm) as afile:
    mesh = afile.load_mesh("mesh2d_per")

# load the initial ensemble (with offsets like attached script)
erank = ecomm.rank
offset = np.concatenate((np.array([0]), np.cumsum(nensemble)))
with fd.CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5",
                       'r', comm=comm) as afile:
    for i in range(nensemble[erank]):
        idx = i + offset[erank]
        psi_chp = afile.load_function(mesh, "psi_checkpoint", idx=idx)
        pv_chp = afile.load_function(mesh, "pv_checkpoint", idx=idx)
        q, psi = jtfilter.ensemble[i][0].subfunctions
        q.interpolate(pv_chp)
        psi.interpolate(psi_chp)

# Likelihood
SD = 0.001
def log_likelihood(y, Y):
    ll = (y - Y)**2 / SD**2 / 2 * fd.dx
    return ll

# Observations
N_obs = 10
psi_VOM = fd.Function(model.VVOM)

# Shared arrays
psi_e_list = []
for m in range(psi_vel.shape[1]):
    psi_e_shared = ndg.SharedArray(partition=nensemble, comm=ecomm)
    psi_e_list.append(psi_e_shared)

psi_shape = psi_vel.shape
if fd.COMM_WORLD.rank == 0:
    psi_e = np.zeros((np.sum(nensemble), N_obs, psi_shape[1]))

# Diagnostics
diagnostics = []
temp_count = []

# Prepare vtk writers per particle
outfile = []
for i in range(nensemble[jtfilter.ensemble_rank]):
    ensemble_idx = sum(nensemble[:jtfilter.ensemble_rank]) + i
    outfile.append(VTKFile(f"../../DA_Results/2DEuler_mixed/vtk_output/{ensemble_idx}_output.pvd",
                           comm=comm))

# Assimilation loop
for k in range(N_obs):
    PETSc.Sys.Print("Assimilation Step", k)
    psi_VOM.dat.data[:] = psi_vel[k, :]

    jtfilter.assimilation_step(psi_VOM, log_likelihood,
                               diagnostics=diagnostics,
                               ess_tol=0.8,
                               taylor_test=False)

    # Save checkpoint for each ensemble at step k
    CG1 = fd.FunctionSpace(model.mesh, "CG", 1)
    qout = fd.Function(CG1, name="asm_particle")
    filename = f"../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_ch_{k}_{erank}.h5"
    with fd.CheckpointFile(filename, 'w', comm=comm) as afile:
        for i in range(nensemble[erank]):
            q, psi = jtfilter.ensemble[i][0].subfunctions
            qout.interpolate(psi)  # save streamfunction
            afile.save_function(qout, idx=i)

    # VTK output for visualization
    for i in range(nensemble[jtfilter.ensemble_rank]):
        particle_vtk = fd.Function(model.Vdg, name='asmlted_particles')
        particle_vtk.interpolate(jtfilter.ensemble[i][0].subfunctions[0])  # PV
        outfile[i].write(particle_vtk, time=k)

    # Shared array update
    for i in range(nensemble[jtfilter.ensemble_rank]):
        obsdata = model.obs().dat.data[:]
        for m in range(psi_vel.shape[1]):
            psi_e_list[m].dlocal[i] = obsdata[m]

    for m in range(psi_vel.shape[1]):
        psi_e_list[m].synchronise()
        if fd.COMM_WORLD.rank == 0:
            psi_e[:, k, m] = psi_e_list[m].data()

    temp_count.append(jtfilter.temper_count)
    Print('temp_count', np.array(temp_count))

# Final save of arrays
if fd.COMM_WORLD.rank == 0:
    np.save("../../DA_Results/2DEuler_mixed/temp_cunt.npy", np.array(temp_count))
    if not nudging:
        np.save("../../DA_Results/2DEuler_mixed/tempjitt_assimilated_Vorticity_ensemble.npy", psi_e)
    if nudging:
        np.save("../../DA_Results/2DEuler_mixed/nudge_assimilated_Vorticity_ensemble.npy", psi_e)
