"""
2D Euler Equation Data Generation for Data Assimilation
======================================================

Creates synthetic observational data for 2D Euler equations:
- Spinup the system to a dynamic equilibrium
- Particle ensemble and truth initialization 
- Forward runs with observation noise


# first conduct uncertainty quantification experiments to determine appropriate noise levels
"""

import firedrake as fd
from firedrake import *
from firedrake.output import VTKFile
from firedrake.petsc import PETSc
from nudging.models.stochastic_mix_euler import Euler_mixSD
import numpy as np
# =============================================================================
# SETUP AND CONFIGURATION
# =============================================================================

# Create output directories
import os
os.makedirs('../../DA_Results/2DEuler_mixed/', exist_ok=True)
os.makedirs('../../DA_Results/2DEuler_mixed/checkpoint_files/', exist_ok=True)

# Initialize VTK output files for visualization
truth_init = VTKFile("../../DA_Results/2DEuler_mixed/paraview_stream/truth_init_salt.pvd")
truth = VTKFile("../../DA_Results/2DEuler_mixed/paraview_stream/truth.pvd")
particle_init = VTKFile("../../DA_Results/2DEuler_mixed/paraview_stream/particle_init.pvd")

Print = PETSc.Sys.Print

# =============================================================================
# PARAMETERS - Exact same structure as original for pickle compatibility
# =============================================================================

params = {}

# Core simulation parameters (same as your original)
nsteps = 5
params["nsteps"] = nsteps

n_x = 64   # partion in x,y direction
params["xpoints"] = n_x

dt = 0.025
params["dt"] = dt



# Model configuration
dw_scale = 1.0
#dw_scale = 0.0
params["noise_scale"] = dw_scale

salt = True    # SALT/SFLT choice
params["salt"] = salt

# ensemble size for data assimilation
nensemble = [2]*30  
params["nensemble"] = nensemble

N_init = 100  # spinup steps

N_obs = 100 # observation steps
params["N_obs"] = N_obs



# =============================================================================
# MODEL SETUP
# =============================================================================

Print("Setting up 2D Euler model...")

# Initialize model with parameters (same as original)
comm = fd.COMM_WORLD
model = Euler_mixSD(
    n_x, 
    nsteps=nsteps,
    dt=dt, 
    noise_scale=dw_scale, 
    salt=salt,      
    lambdas=True     # Use lambda for nudging
)

model.setup(comm=fd.COMM_WORLD)
mesh = model.mesh

# Symbolic spatial coordinate for initial conditions
x = fd.SpatialCoordinate(mesh)

# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def gradperp(u):
    """Compute perpendicular gradient: ∇⊥u = (-∂u/∂y, ∂u/∂x)"""
    return fd.as_vector((-u.dx(1), u.dx(0)))

def extract_observations(model):
    """Extract observations at specified points and return as numpy array"""
    psi_VOM = model.obs()
    psi_VOM_out = fd.Function(model.VVOM_out)
    psi_VOM_out.interpolate(psi_VOM)
    print('psi_VOM_out.dat.data.shape', psi_VOM_out.dat.data.shape)
    return psi_VOM_out.dat.data_ro.copy()

def compute_kinetic_energy(model, mesh):
    """Compute kinetic energy from velocity field u = ∇⊥ψ"""
    Vu = fd.VectorFunctionSpace(mesh, "DQ", 0)  # DQ elements for velocity
    v = fd.Function(Vu, name="velocity")
    v.project(gradperp(model.qpsi0[1]))  # u = gradperp(streamfunction)
    return 0.5 * fd.norm(v)

def compute_courant_number(model, mesh, dt):
    Vu = fd.VectorFunctionSpace(mesh, "DQ", 0)
    u = fd.Function(Vu, name="velocity")
    u.project(gradperp(model.qpsi0[1]))
    V0 = fd.FunctionSpace(mesh, "DG", 0)
    C = fd.Function(V0)
    C.project((fd.sqrt(fd.dot(u, u))) * dt / fd.CellSize(mesh))
    return fd.assemble(C)

# =============================================================================
# INITIAL CONDITIONS
# =============================================================================

# Allocate state vector
X0_truth = model.allocate()
q0, psi0 = X0_truth[0].subfunctions

# Multi-scale initial vorticity field
sin, cos, pi = fd.sin, fd.cos, fd.pi
initial_vorticity = (
    sin(8*pi*x[0]) * sin(8*pi*x[1]) +           # Primary wave mode
    0.4 * cos(6*pi*x[0]) * cos(6*pi*x[1]) +     # Secondary mode
    0.3 * cos(10*pi*x[0]) * cos(4*pi*x[1]) +    # High frequency component
    0.02 * sin(2*pi*x[0]) +                     # Large scale x-variation
    0.02 * sin(2*pi*x[1])                       # Large scale y-variation
)

q0.interpolate(initial_vorticity)

# =============================================================================
# PHASE 1: TRUTH INITIALIZATION
# =============================================================================

Print("Finding an initial state.")
Print("Phase 1: Spinup before initialization")

# Storage for energy evolution
u_energy = []

# Initialize observation storage
psi_true_template = extract_observations(model)
psi_init = np.zeros(np.size(psi_true_template))

# Run model forward to generate initial state
for i in fd.ProgressBar("Truth initialization").iter(range(N_init)):
    
    # Add stochastic forcing and advance one step
    model.randomize(X0_truth)
    model.run(X0_truth, X0_truth)

    Cmax = compute_courant_number(model, mesh, dt)
    print(f"Courant number: {Cmax.dat.data_ro.max():.3f}")
    
    # Extract state variables for output
    q, psi = model.qpsi1.subfunctions
    dU = model.dU_3  # Noise realization
    
    # Rename for clear paraview output
    q.rename("Vorticity")
    psi.rename("Stream Function") 
    dU.rename("Noise Variable")
    
    # Compute and store kinetic energy
    energy = compute_kinetic_energy(model, mesh)
    u_energy.append(energy)
    
    # Write to paraview for visualization
    truth_init.write(q, psi, dU)
    
    #Store final initialization data
    if i == N_init - 1:
        psi_init = extract_observations(model)
        if comm.rank == 0:
            np.save("../../DA_Results/2DEuler_mixed/psi_init.npy", psi_init)

# Save energy evolution
if comm.rank == 0:
    with open("../../DA_Results/2DEuler_mixed/energy_evolution.txt", "w") as f:
        f.write("# Step\tEnergy\n")
        for i, energy in enumerate(u_energy):
            f.write(f"{i}\t{energy:.6e}\n")

# =============================================================================
# PHASE 2: PARTICLE ENSEMBLE INITIALIZATION  
# =============================================================================

Print("Phase 2: Particle ensemble initialization")

# Setup checkpoint functions
Vcg = fd.FunctionSpace(mesh, "CG", 1)  # Continuous Galerkin for streamfunction
Vdg = fd.FunctionSpace(mesh, "DQ", 1)  # Discontinuous Galerkin for vorticity
psi_chp = fd.Function(Vcg, name="psi_checkpoint")
pv_chp = fd.Function(Vdg, name="pv_checkpoint") 

# Storage for all particle initializations
total_particles = sum(nensemble) + 1  # +1 for truth
psi_particle_init = np.zeros((total_particles, len(psi_init)))

X_particle = model.allocate()
particle_count = 0

ndump = 10  # dump data

# Generate ensemble particles + truth
with fd.CheckpointFile("../../DA_Results/2DEuler_mixed/checkpoint_files/ensemble_init.h5", 'w') as afile:
    
    for i in fd.ProgressBar("").iter(range(total_particles)):
        
        if i < sum(nensemble):
            Print("Initlisation of  particles")
        else:
            Print("Initlisation of truth")
        
        # Start all particles from same initial condition
        X_particle[0].assign(X0_truth[0])
        
        # Run forward to decorrelate particles
        for j in range(ndump):
            model.randomize(X_particle)
            model.run(X_particle, X_particle)
        
        # Extract final state
        q, psi = model.qpsi1.subfunctions
        q.rename("Vorticity")
        psi.rename("Stream Function")
        print('psi cts', psi.dat.data.shape)
        print('q cts', q.dat.data.shape)
        # Save to checkpoint file
        psi_chp.interpolate(psi)
        pv_chp.interpolate(q)
        afile.save_function(psi_chp, idx=i)
        afile.save_function(pv_chp, idx=i)
        
        # Write visualization
        particle_init.write(q, psi)
        
        # Store observations for this particle
        psi_particle = extract_observations(model)
        if comm.rank == 0:
            psi_particle_init[particle_count, :] = psi_particle
            particle_count += 1

# Save particle initialization data
if comm.rank == 0:
    np.save("../../DA_Results/2DEuler_mixed/psi_particle_init.npy", psi_particle_init)


# # Save parameters to pickle file
# import pickle
# with open('params.pickle', 'wb') as handle:
#     pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)







# =============================================================================
# PHASE 3: OBSERVATIONAL DATA GENERATION
# =============================================================================

Print("Generating the observational data.")
Print("Phase 3: Observational data generation")

# Initialize truth from final particle state
X_truth = model.allocate()
X_truth[0].assign(X_particle[0])

# Storage arrays for observations
n_obs_points = len(psi_init)
psi_true_all = np.zeros((N_obs, n_obs_points))
psi_obs_all = np.zeros((N_obs, n_obs_points))


noise_sd = 0.03  # Observation noise variance
params["noise_sd"] = noise_sd

# Generate time series of observations
for i in fd.ProgressBar("").iter(range(N_obs)):
    
    # Advance truth with stochastic forcing
    model.randomize(X_truth)
    model.run(X_truth, X_truth)
    
    # Extract state for visualization
    q, psi = model.qpsi1.subfunctions
    q.rename("Vorticity")
    psi.rename("Stream Function")
    truth.write(q, psi)
    
    # Extract true observations
    psi_true = extract_observations(model)
    
    if comm.rank == 0:
        # Store true observations
        psi_true_all[i, :] = psi_true
        
        # # Add multiplicative observation noise
        # determine number of observation points from the extracted vector
        obs_points = psi_true.size            # or use n_obs_points which is len(psi_init)
        psi_noise = np.random.normal(0.0, noise_sd, obs_points)  # mean=0, sd = sqrt(noise_var)
        
        # Scale noise relative to signal magnitude (preserves boundary conditions)
        # psi_max = np.abs(psi_true).max()
        psi_obs = psi_true +  psi_noise
        
        psi_obs_all[i, :] = psi_obs

# Save observational datasets
if comm.rank == 0:
    np.save("../../DA_Results/2DEuler_mixed/psi_true_data.npy", psi_true_all)
    np.save("../../DA_Results/2DEuler_mixed/psi_obs_data.npy", psi_obs_all)




# Save parameters to pickle file
import pickle
with open('params.pickle', 'wb') as handle:
    pickle.dump(params, handle, protocol=pickle.HIGHEST_PROTOCOL)
