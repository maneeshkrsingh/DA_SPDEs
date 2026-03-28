from firedrake import dx
from nudging import LSDEModel, \
    jittertemp_filter, base_diagnostic, Stage
from nudging.resampling import residual_resampling
import numpy as np
from firedrake.petsc import PETSc
Print = PETSc.Sys.Print
print = PETSc.Sys.Print

# Fixed parameters
T = 1.
nsteps = 5
dt = T/nsteps
A = 1.
D = 1.0
p_per_rank = 10
nranks = 30
nensemble = [p_per_rank]*nranks
S = 0.1  # observation noise standard deviation
y0 = -0.05563397349186569

# Number of repetitions
num_experiments = 50

# Generate random seeds for experiments
np.random.seed(42)
model_seeds = np.random.randint(1000, 1000000, size=num_experiments)
resampler_seeds = np.random.randint(1000, 1000000, size=num_experiments)

# Storage for estimators
mean_estimators = []
variance_estimators = []

def log_likelihood(y, Y):
    ll = (y-Y)**2/S**2/2*dx
    return ll

# Compute analytical (true) values
c = 0.0
d = D**2/2/A
sigsq = D**2/2/A*(1 - np.exp(-2*A*T))
Sigsq = sigsq + np.exp(-2*A*T)*d
true_mean = (Sigsq*y0 + np.exp(-A*T)*S**2*c)/(Sigsq + S**2)
true_variance = Sigsq*S**2/(Sigsq + S**2)

Print("="*80)
Print(f"TRUE (ANALYTICAL) VALUES:")
Print(f"  Mean:     {true_mean:.8f}")
Print(f"  Variance: {true_variance:.8f}")
Print("="*80)
Print(f"\nRunning {num_experiments} experiments with different random seeds...")
Print("="*80)

# Run experiments - create NEW model and filter each time
for exp_num in range(num_experiments):
    model_seed = model_seeds[exp_num]
    resampler_seed = resampler_seeds[exp_num]
    
    if exp_num % 10 == 0:
        Print(f"\nExperiment {exp_num+1}/{num_experiments} (model_seed={model_seed}, resampler_seed={resampler_seed})")
    
    # Create NEW model with specific seed
    model = LSDEModel(A=A, D=D, nsteps=nsteps, dt=dt, lambdas=True, seed=model_seed)
    
    # Create NEW filter
    myfilter = jittertemp_filter(n_jitt=0, delta=0.025,
                                 verbose=0, MALA=False,
                                 visualise_tape=False, nudging=True, sigma=0.01)
    myfilter.setup(nensemble=nensemble, model=model, residual=False)
    
    # Set observation
    y = model.obs()
    y.dat.data[:] = y0
    
    # Prepare initial ensemble - THIS IS YOUR ORIGINAL WORKING CODE
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
    try:
        myfilter.assimilation_step(y, log_likelihood,
                                   diagnostics=diagnostics,
                                   ess_tol=-9000.8,
                                   taylor_test=False,
                                   tao_params=tao_params)
        
        # Collect results from this experiment
        if myfilter.subcommunicators.global_comm.rank == 0:
            before, descriptors = nolambdasamples.get_archive()
            after, descriptors = nudgingsamples.get_archive()
            resampled, descriptors = resamplingsamples.get_archive()
            
            # Compute estimators for this experiment
            estimated_mean = np.mean(resampled)
            estimated_variance = np.var(resampled)
            
            mean_estimators.append(estimated_mean)
            variance_estimators.append(estimated_variance)
        
        # Clean up to avoid MPI issues
        del model
        del myfilter
        PETSc.garbage_cleanup(PETSc.COMM_WORLD)
        
    except Exception as e:
        Print(f"ERROR in experiment {exp_num+1}: {e}")
        Print("Continuing to next experiment...")
        continue

# Compute final statistics on rank 0
if len(mean_estimators) > 0:
    mean_estimators = np.array(mean_estimators)
    variance_estimators = np.array(variance_estimators)
    
    mean_of_mean_estimators = np.mean(mean_estimators)
    mean_of_variance_estimators = np.mean(variance_estimators)
    
    diff_mean = true_mean - mean_of_mean_estimators
    diff_variance = true_variance - mean_of_variance_estimators
    
    Print("\n" + "="*80)
    Print("FINAL RESULTS OVER ALL EXPERIMENTS")
    Print("="*80)
    Print(f"\nNumber of successful experiments: {len(mean_estimators)}")
    
    Print("\n" + "-"*80)
    Print("MEAN ESTIMATOR:")
    Print("-"*80)
    Print(f"  True mean:                       {true_mean:.8f}")
    Print(f"  Mean of estimators:              {mean_of_mean_estimators:.8f}")
    Print(f"  Difference (true - estimated):   {diff_mean:.8f}")
    
    Print("\n" + "-"*80)
    Print("VARIANCE ESTIMATOR:")
    Print("-"*80)
    Print(f"  True variance:                   {true_variance:.8f}")
    Print(f"  Mean of estimators:              {mean_of_variance_estimators:.8f}")
    Print(f"  Difference (true - estimated):   {diff_variance:.8f}")
    
    Print("\n" + "="*80)
    
    # Save results
    results_dict = {
        'experiment_number': np.arange(len(mean_estimators)),
        'model_seed': model_seeds[:len(mean_estimators)],
        'resampler_seed': resampler_seeds[:len(mean_estimators)],
        'mean_estimator': mean_estimators,
        'variance_estimator': variance_estimators,
    }
    
    import pandas as pd
    df = pd.DataFrame(results_dict)
    df.to_csv('estimator_results.csv', index=False)
    Print("\nDetailed results saved to 'estimator_results.csv'")
    
    summary = {
        'Metric': ['True Value', 'Mean of Estimators', 'Difference (True - Estimated)'],
        'Mean': [true_mean, mean_of_mean_estimators, diff_mean],
        'Variance': [true_variance, mean_of_variance_estimators, diff_variance]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('estimator_summary.csv', index=False)
    Print("Summary saved to 'estimator_summary.csv'")
    
    Print("\n" + "="*80)
    Print("ANALYSIS COMPLETE")
    Print("="*80)