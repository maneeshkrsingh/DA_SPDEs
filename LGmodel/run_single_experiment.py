from firedrake import dx
from nudging import LSDEModel, \
    jittertemp_filter, base_diagnostic, Stage
from nudging.resampling import residual_resampling
import numpy as np
from firedrake.petsc import PETSc
import sys
import os

Print = PETSc.Sys.Print

# Get seeds from command line arguments
model_seed = int(sys.argv[1])
resampler_seed = int(sys.argv[2])
exp_num = int(sys.argv[3])
output_dir = sys.argv[4]

# Fixed parameters
T = 1.
nsteps = 5
dt = T/nsteps
A = 1.
D = 1.0
p_per_rank = 12
nranks = 25
nensemble = [p_per_rank]*nranks
S = 0.1
y0 = -0.05563397349186569

def log_likelihood(y, Y):
    ll = (y-Y)**2/S**2/2*dx
    return ll

# Create model
model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=True, seed=model_seed)

# Create filter
myfilter = jittertemp_filter(n_jitt=0, delta=0.15,
                             verbose=0, MALA=False,
                             visualise_tape=False, nudging=False, sigma=0.01)
myfilter.setup(nensemble=nensemble, model=model, residual=False)

# Set observation
y = model.obs()
y.dat.data[:] = y0

# Prepare initial ensemble
c = 0.
d = D**2/2/A
for i in range(nensemble[myfilter.ensemble_rank]):
    dx0 = model.rg.normal(model.R, c, d)
    u = myfilter.ensemble[i][0]
    u.assign(dx0)

# Set up diagnostics
class samples(base_diagnostic):
    def compute_diagnostic(self, particle):
        model.u.assign(particle[0])
        return model.obs().dat.data[0]

nolambdasamples = samples(Stage.WITHOUT_LAMBDAS,
                          myfilter.subcommunicators,
                          nensemble)
nudgingsamples = samples(Stage.AFTER_NUDGING,
                         myfilter.subcommunicators,
                         nensemble)
resamplingsamples = samples(Stage.AFTER_ASSIMILATION_STEP,
                            myfilter.subcommunicators,
                            nensemble)
diagnostics = [nudgingsamples, resamplingsamples, nolambdasamples]

tao_params = {
    "tao_type": "lmvm",
    "tao_monitor": None,
    "tao_converged_reason": None,
    "tao_gatol": 1.0e-4,
    "tao_grtol": 1.0e-50,
    "tao_gttol": 1.0e-5,
}

# Set resampler seed
myfilter.resampler = residual_resampling(seed=resampler_seed, residual=False)

# Run assimilation
myfilter.assimilation_step(y, log_likelihood,
                           diagnostics=diagnostics,
                           ess_tol=-9900.8,
                           taylor_test=False,
                           tao_params=tao_params)

# Save results from this experiment
if myfilter.subcommunicators.global_comm.rank == 0:
    os.makedirs(output_dir, exist_ok=True)

    before, descriptors = nolambdasamples.get_archive()
    after, descriptors = nudgingsamples.get_archive()
    resampled, descriptors = resamplingsamples.get_archive()

    estimated_mean = np.mean(resampled)
    estimated_variance = np.var(resampled)

    result = {
        'experiment': exp_num,
        'model_seed': model_seed,
        'resampler_seed': resampler_seed,
        'mean_estimator': estimated_mean,
        'variance_estimator': estimated_variance
    }

    save_path = os.path.join(output_dir, f'experiment_{exp_num}.npy')
    np.save(save_path, result)
    Print(f"Experiment {exp_num}: mean={estimated_mean:.8f}, var={estimated_variance:.8f}")
    #Print(f"Saved to {save_path}")