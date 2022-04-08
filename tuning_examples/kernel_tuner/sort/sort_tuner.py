#!/usr/bin/env python
import argparse
from collections import OrderedDict

from kernel_tuner import tune_kernel
from numba import cuda

from common import store_BAT_results

def tune(size, strategy, test=False):
    """ Function to setup tunable parameters and tune the sort benchmark """

    # Use host code in combination with CUDA kernel
    kernel_files = ['sort_host.cu', '../../../src/kernels/sort/sort_kernel.cu']

    gpu = cuda.get_current_device()
    max_size = gpu.MAX_THREADS_PER_BLOCK
    # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK` and over 16
    block_sizes = list(filter(lambda x: x <= max_size, [2**i for i in range(4, 11)]))

    # Add parameters to tune
    tune_params = OrderedDict()
    tune_params["LOOP_UNROLL_LSB"] = [0, 1]
    tune_params["LOOP_UNROLL_LOCAL_MEMORY"] = [0, 1]
    tune_params["SCAN_DATA_SIZE"] = [2, 4, 8]  # vector width of input data
    tune_params["SORT_DATA_SIZE"] = [2, 4, 8]  # number of elements per thread
    tune_params["SCAN_BLOCK_SIZE"] = block_sizes
    tune_params["SORT_BLOCK_SIZE"] = block_sizes
    tune_params["INLINE_LSB"] = [0, 1]
    tune_params["INLINE_SCAN"] = [0, 1]
    tune_params["INLINE_LOCAL_MEMORY"] = [0, 1]

    if test:
        tune_params["LOOP_UNROLL_LSB"] = [0]
        tune_params["LOOP_UNROLL_LOCAL_MEMORY"] = [0]
        tune_params["SCAN_DATA_SIZE"] = [2, 4, 8]  # vector width of input data
        tune_params["SORT_DATA_SIZE"] = [2]  # number of elements per thread
        tune_params["SCAN_BLOCK_SIZE"] = [block_sizes[0]]
        tune_params["SORT_BLOCK_SIZE"] = [block_sizes[0]]
        tune_params["INLINE_LSB"] = [0]
        tune_params["INLINE_SCAN"] = [0]
        tune_params["INLINE_LOCAL_MEMORY"] = [0]

    # Constraint to ensure not attempting to use too much shared memory
    # 4 is the size of uints and 2 is because shared memory is used for both keys and values in the "reorderData" function
    # 16 * 2 is also added due to two other shared memory uint arrays used for offsets
    gpu = cuda.get_current_device()
    available_shared_memory = gpu.MAX_SHARED_MEMORY_PER_BLOCK

    # Constraint for block sizes and data sizes
    constraint = ["(SCAN_BLOCK_SIZE / SORT_BLOCK_SIZE) == (SORT_DATA_SIZE / SCAN_DATA_SIZE)",
                 f"((SCAN_BLOCK_SIZE * SCAN_DATA_SIZE * 4 * 2) + (4 * 16 * 2)) <= {available_shared_memory}"]

    strategy_options = {}
    if strategy == "genetic_algorithm":
        strategy_options = {"maxiter": 50, "popsize": 10}

    # Tune all kernels and correctness verify by throwing error if verification failed
    tuning_results = tune_kernel("sort", kernel_files, size, [], tune_params, strategy=strategy,
                                lang="C", restrictions=constraint,
                                compiler_options=["-I ../../../src/kernels/sort/", f"-DPROBLEM_SIZE={input_problem_size}"],
                                iterations=2, strategy_options=strategy_options)

    return tuning_results, tune_params


if __name__ == "__main__":

    # Setup CLI parser
    parser = argparse.ArgumentParser(description="Sort tuner")
    parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
    parser.add_argument("--technique", "-t", type=str, default="brute_force", help="tuning technique to use for the benchmark (e.g.: annealing)")
    parser.add_argument("--test", "-T", type=int, default=0, help="run only a test and not the full search space (0 or 1, default 0)")
    arguments = parser.parse_args()

    # Problem sizes used in the SHOC benchmark
    problem_sizes = [1, 8, 48, 96]
    input_problem_size = arguments.size
    size = int((problem_sizes[input_problem_size - 1] * 1024 * 1024) / 4) # 4 = sizeof(uint)

    tuning_results, tune_params = tune(size, arguments.technique, arguments.test)

    store_BAT_results("sort", tuning_results, input_problem_size, arguments.technique, tune_params)
