import numpy
from kernel_tuner import tune_kernel, run_kernel
from numba import cuda
import json


# Problem sizes used in the SHOC benchmark
problem_sizes = [16384, 32768, 65536, 131072, 262144, 524288, 1048576, 2097152, 4194304]

gpu = cuda.get_current_device()
max_block_size = gpu.MAX_THREADS_PER_BLOCK

# TODO use different problem sizes from input
size = problem_sizes[8]

# Use host code in combination with CUDA kernel
kernel_files = ['triad_host.cu', '../../../src/kernels/triad/triad_kernel.cu']

# Add parameters to tune
tune_params = dict()
tune_params["BLOCK_SIZE"] = [i for i in range(1, max_block_size + 1)] # Range: [1, ..., max_block_size]
tune_params["WORK_PER_THREAD"] = [i for i in range(1, 11)] # Range: [1, ..., 10]
tune_params["LOOP_UNROLL_TRIAD"] = [0, 1]
tune_params["PRECISION"] = [32, 64]

# Tune kernel and correctness verify by throwing error if verification failed
tuning_results = tune_kernel("triad_host", kernel_files, size, [], tune_params, lang="C", block_size_names=["BLOCK_SIZE"], compiler_options=["-I ../../../src/kernels/triad/"])

# Save the results as a JSON file
with open("triad-results.json", 'w') as f:
    json.dump(tuning_results, f)

# Get the best configuration
best_parameter_config = min(tuning_results[0], key=lambda x: x['time'])
best_parameters = dict()

# Filter out parameters from results
for k, v in best_parameter_config.items():
    if k not in tune_params:
        continue

    best_parameters[k] = v

# Save the best results as a JSON file
with open("best-triad-results.json", 'w') as f:
    json.dump(best_parameters, f)
