#!/usr/bin/env python
import os
import argparse
from collections import OrderedDict

import numpy as np
from kernel_tuner import tune_kernel

from common import store_BAT_results, get_device_info

dir_path = os.path.dirname(os.path.realpath(__file__))

kernel_file = dir_path + "/../../../src/kernels/sort/sort_kernel.cu"
data_dir = dir_path + "/../../../src/kernels/sort/data/scan/"

def read_input_data(input_problem_size, size):

    def read_txt_file(filename):
        with open(filename, "r") as fh:
            holder = np.zeros(size, dtype=np.int32)
            data = np.array([int(i) for i in fh.read().strip().split("\n")]).astype(np.int32)
            holder[:data.size] = data
            return holder

    block_sums = read_txt_file(f"{data_dir}{input_problem_size}-blockSums")
    scan_input = read_txt_file(f"{data_dir}{input_problem_size}-scanInput")
    scan_output = read_txt_file(f"{data_dir}{input_problem_size}-scanOutput")

    return [scan_output, scan_input, block_sums]


def tune(input_problem_size, strategy, test=False):
    """ Function to setup tunable parameters and tune the sort benchmark """

    # Only tune CUDA kernel
    problem_sizes = [1, 8, 48, 96]
    size = int((problem_sizes[input_problem_size - 1] * 1024 * 1024) / 4) # 4 = sizeof(uint)

    def problem_size_func(p):                                        #matching line no. in programs/sort/sort.cu
        reorderFindGlobalWorkSize = size // p['SCAN_DATA_SIZE']                              #L217
        reorderBlocks = reorderFindGlobalWorkSize // p['SCAN_BLOCK_SIZE']                    #L222
        numElements = 16 * reorderBlocks                                                     #L232
        numBlocks = int(np.ceil(numElements / (p['SORT_DATA_SIZE'] * p['SCAN_BLOCK_SIZE']))) #L242
        grid = (numBlocks, 1, 1)                                                             #L248
        return grid

    problem_size = problem_size_func
    grid_div_x = []

    block_size_names = ["SCAN_BLOCK_SIZE"]

    args = read_input_data(input_problem_size, size) + [np.int32(size), np.uint8(1), np.uint8(1)]

    gpu = get_device_info(0)
    max_size = gpu["max_threads"]
    # Using 2^i values less than `gpu.MAX_THREADS_PER_BLOCK` and over 16
    block_sizes = list(filter(lambda x: x <= max_size, [2**i for i in range(5, 11)]))

    # Add parameters to tune
    tune_params = OrderedDict()
    tune_params["LOOP_UNROLL_LOCAL_MEMORY"] = [0, 1]
    tune_params["SCAN_DATA_SIZE"] = [2, 4, 8]  # vector width of input data
    tune_params["SORT_DATA_SIZE"] = [2, 4, 8]  # number of elements per thread
    tune_params["SCAN_BLOCK_SIZE"] = block_sizes
    tune_params["SORT_BLOCK_SIZE"] = block_sizes
    tune_params["INLINE_LOCAL_MEMORY"] = [0, 1]

    if test:
        tune_params["LOOP_UNROLL_LOCAL_MEMORY"] = [0]
        tune_params["SCAN_DATA_SIZE"] = [2, 4, 8]  # vector width of input data
        tune_params["SORT_DATA_SIZE"] = [2]  # number of elements per thread
        tune_params["SCAN_BLOCK_SIZE"] = [block_sizes[-1]]
        tune_params["SORT_BLOCK_SIZE"] = [block_sizes[-1]]
        tune_params["INLINE_LOCAL_MEMORY"] = [0]

    # Constraint to ensure not attempting to use too much shared memory
    # 4 is the size of uints and 2 is because shared memory is used for both keys and values in the "reorderData" function
    # 16 * 2 is also added due to two other shared memory uint arrays used for offsets
    available_shared_memory = gpu["max_shared_memory"]

    # Constraint for block sizes and data sizes
    constraint = ["(SCAN_BLOCK_SIZE / SORT_BLOCK_SIZE) == (SORT_DATA_SIZE / SCAN_DATA_SIZE)",
                 f"((SCAN_BLOCK_SIZE * SCAN_DATA_SIZE * 4 * 2) + (4 * 16 * 2)) <= {available_shared_memory}"]

    strategy_options = {}
    if strategy == "genetic_algorithm":
        strategy_options = {"maxiter": 50, "popsize": 10}

    # Tune all kernels and correctness verify by throwing error if verification failed
    tuning_results = tune_kernel("scan", kernel_file, problem_size, args, tune_params, strategy=strategy,
                                restrictions=constraint,
                                grid_div_x=grid_div_x, block_size_names=block_size_names,
                                lang="cupy",
                                iterations=7, strategy_options=strategy_options)

    return tuning_results, tune_params


if __name__ == "__main__":

    # Setup CLI parser
    parser = argparse.ArgumentParser(description="Sort tuner")
    parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark (e.g.: 2)")
    parser.add_argument("--technique", "-t", type=str, default="brute_force", help="tuning technique to use for the benchmark (e.g.: annealing)")
    parser.add_argument("--test", "-T", type=int, default=0, help="run only a test and not the full search space (0 or 1, default 0)")
    arguments = parser.parse_args()

    # Problem sizes used in the SHOC benchmark
    input_problem_size = arguments.size

    tuning_results, tune_params = tune(arguments.size, arguments.technique, arguments.test)

    store_BAT_results("sort", tuning_results, arguments.size, arguments.technique, tune_params)
