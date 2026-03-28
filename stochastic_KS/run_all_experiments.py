"""
Run multiple KS DA experiments with different seeds.
Usage: python run_all_experiments.py <filter_type> [num_experiments]

Examples:
    python run_all_experiments.py tempjitt 1    # quick sanity test
    python run_all_experiments.py nudge 50      # full study

Run from: ~/DA_SPDEs/stochastic_KS/
Firedrake venv must be activated before running this script.
"""
import subprocess
import os
import sys
import numpy as np

filter_type = sys.argv[1] if len(sys.argv) > 1 else 'tempjitt'
num_experiments = int(sys.argv[2]) if len(sys.argv) > 2 else 50

output_dir = '../../DA_KS/multirun_results'
os.makedirs(output_dir, exist_ok=True)

# Generate reproducible seeds
rng = np.random.default_rng(42)
model_seeds = rng.integers(10000, 99999, size=num_experiments)
resampler_seeds = rng.integers(10000, 99999, size=num_experiments)

# Save seed table for reproducibility
seed_file = os.path.join(output_dir, f"seeds_{filter_type}.npy")
np.save(seed_file, np.column_stack((model_seeds, resampler_seeds)))

failed = []
start_from = int(sys.argv[3]) if len(sys.argv) > 3 else 0
for exp_num in range(start_from, num_experiments):
    print(f"\n{'='*60}")
    print(f"Experiment {exp_num+1}/{num_experiments} "
          f"[model_seed={model_seeds[exp_num]}, "
          f"resampler_seed={resampler_seeds[exp_num]}]")
    print(f"{'='*60}\n")

    cmd = [
        'mpiexec', '-n', '30',
        'python', 'run_single_experiment.py',
        str(model_seeds[exp_num]),
        str(resampler_seeds[exp_num]),
        str(exp_num),
        output_dir,
        filter_type,
    ]

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR in experiment {exp_num}: {e}")
        failed.append(exp_num)
        continue

print(f"\n{'='*60}")
print(f"Done. {num_experiments - len(failed)}/{num_experiments} succeeded.")
if failed:
    print(f"Failed experiments: {failed}")
print(f"Results in {output_dir}/{filter_type}/")
print(f"{'='*60}")
