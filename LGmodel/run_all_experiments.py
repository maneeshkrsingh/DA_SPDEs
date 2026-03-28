import numpy as np
import subprocess
import os
from glob import glob
import pandas as pd

# Number of experiments
num_experiments = 1000

# Generate seeds
np.random.seed(42)
model_seeds = np.random.randint(1000, 1000000, size=num_experiments)
resampler_seeds = np.random.randint(1000, 1000000, size=num_experiments)

# Compute analytical values
T = 1.
A = 1.
D = 1.0
S = 0.1
y0 = -0.05563397349186569

c = 0.0
d = D**2/2/A
sigsq = D**2/2/A*(1 - np.exp(-2*A*T))
Sigsq = sigsq + np.exp(-2*A*T)*d
true_mean = (Sigsq*y0 + np.exp(-A*T)*S**2*c)/(Sigsq + S**2)
true_variance = Sigsq*S**2/(Sigsq + S**2)

print("="*80)
print(f"TRUE (ANALYTICAL) VALUES:")
print(f"  Mean:     {true_mean:.8f}")
print(f"  Variance: {true_variance:.8f}")
print("="*80)
print(f"\nRunning {num_experiments} experiments...")
print("="*80)

# Output folder
output_dir = "results"
os.makedirs(output_dir, exist_ok=True)

# Clean up old experiment files in results folder
for f in glob(os.path.join(output_dir, 'experiment_*.npy')):
    os.remove(f)

# Run each experiment as separate process
for exp_num in range(num_experiments):
    model_seed = model_seeds[exp_num]
    resampler_seed = resampler_seeds[exp_num]

    if exp_num % 10 == 0:
        print(f"\nStarting experiment {exp_num+1}/{num_experiments}")

    cmd = [
        'mpiexec', '-n', '25',
        'python', 'run_single_experiment.py',
        str(model_seed), str(resampler_seed), str(exp_num), output_dir
    ]

    try:
        subprocess.run(cmd, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in experiment {exp_num}: {e}")
        continue

# Collect all results from results folder
print("\n" + "="*80)
print("Collecting results...")
print("="*80)

mean_estimators = []
variance_estimators = []
successful_experiments = []

for exp_num in range(num_experiments):
    filename = os.path.join(output_dir, f'experiment_{exp_num}.npy')
    if os.path.exists(filename):
        result = np.load(filename, allow_pickle=True).item()
        mean_estimators.append(result['mean_estimator'])
        variance_estimators.append(result['variance_estimator'])
        successful_experiments.append(exp_num)

if len(mean_estimators) > 0:
    mean_estimators = np.array(mean_estimators)
    variance_estimators = np.array(variance_estimators)

    mean_of_mean_estimators = np.mean(mean_estimators)
    mean_of_variance_estimators = np.mean(variance_estimators)

    diff_mean = true_mean - mean_of_mean_estimators
    diff_variance = true_variance - mean_of_variance_estimators

    print("\n" + "="*80)
    print("FINAL RESULTS OVER ALL EXPERIMENTS")
    print("="*80)
    print(f"\nNumber of successful experiments: {len(mean_estimators)}/{num_experiments}")

    print("\n" + "-"*80)
    print("MEAN ESTIMATOR:")
    print("-"*80)
    print(f"  True mean:                       {true_mean:.8f}")
    print(f"  Mean of estimators:              {mean_of_mean_estimators:.8f}")
    print(f"  Difference (true - estimated):   {diff_mean:.8f}")

    print("\n" + "-"*80)
    print("VARIANCE ESTIMATOR:")
    print("-"*80)
    print(f"  True variance:                   {true_variance:.8f}")
    print(f"  Mean of estimators:              {mean_of_variance_estimators:.8f}")
    print(f"  Difference (true - estimated):   {diff_variance:.8f}")

    print("\n" + "="*80)

    results_dict = {
        'experiment_number': successful_experiments,
        'model_seed': model_seeds[successful_experiments],
        'resampler_seed': resampler_seeds[successful_experiments],
        'mean_estimator': mean_estimators,
        'variance_estimator': variance_estimators,
    }

    df = pd.DataFrame(results_dict)
    df.to_csv('estimator_results.csv', index=False)
    print("\nDetailed results saved to 'estimator_results.csv'")

    summary = {
        'Metric': ['True Value', 'Mean of Estimators', 'Difference (True - Estimated)'],
        'Mean': [true_mean, mean_of_mean_estimators, diff_mean],
        'Variance': [true_variance, mean_of_variance_estimators, diff_variance]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv('estimator_summary.csv', index=False)
    print("Summary saved to 'estimator_summary.csv'")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
else:
    print("ERROR: No experiments completed successfully!")