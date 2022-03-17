#!/usr/bin/env python

import os
import subprocess
import argparse
from helpers import get_subdirectories, retrieve_benchmark_config, retrieve_parameter_results, \
    copy_benchmark_result_files, print_helpers, benchmark_dir


# By default benchmark=None and auto_tuner=None. Either of them is required to be specified in order to
# run the benchmark
def assert_no_errors(benchmark_name, auto_tuner, ):
    if not benchmark_name and not auto_tuner:
        print(f"{print_helpers['error']} You have to specify at least one of `benchmark_name` and `auto_tuner`")
        return


def run_benchmark(benchmark_name=None, auto_tuner=None, verbose=False,
                  start_directory=benchmark_dir, problem_size=1, tuning_technique=""):
    print(f"{print_helpers['info']} "
          f"Running {'benchmark `' + benchmark_name + '`' if not benchmark_name is None else 'all benchmarks'} "
          f"for {'`' + auto_tuner + '`' if not auto_tuner is None else 'all auto-tuners'}")

    auto_tuner_dirs = get_subdirectories(start_directory)

    # Filter out auto-tuner directory if specified auto-tuner
    if auto_tuner is not None:
        auto_tuner_dirs = [directory for directory in auto_tuner_dirs if os.path.basename(directory) == auto_tuner]

    # If no auto-tuner directories are found. Can happen if specified other start directory or auto-tuner with
    # invalid name
    if len(auto_tuner_dirs) == 0:
        print(f"{print_helpers['error']} No auto-tuner directories were found with the name `{auto_tuner}`")
        return

    found_benchmarks = False

    # Find all benchmarks for all auto-tuners
    for directory in auto_tuner_dirs:
        auto_tuner_name = os.path.basename(directory)
        print(
            f"{print_helpers['info']} Finding benchmark{'s' if benchmark_name is None else ' `' + benchmark_name + '`'} for `{auto_tuner_name}`")

        # If no name => run all benchmark dirs, otherwise run the selected one
        if benchmark_name is None:
            benchmark_dirs = get_subdirectories(directory)
        else:
            benchmark_dirs = [os.path.join(directory, benchmark_name)]

        # Run all benchmarks for current auto-tuner
        for current_benchmark_dir in benchmark_dirs:
            current_benchmark = os.path.basename(current_benchmark_dir)
            benchmark_config = retrieve_benchmark_config(current_benchmark_dir)

            # Go to next directory if no benchmark found
            if benchmark_config is None:
                print(
                    f"{print_helpers['error']} No benchmark found for `{os.path.basename(current_benchmark_dir)}` in `{auto_tuner_name}`")
                continue

            found_benchmarks = True
            build_successful = True

            # Run the `build` commands in the benchmark directory if its present, array and not empty
            if "build" in benchmark_config and isinstance(benchmark_config["build"], list) and len(
                    benchmark_config["build"]) > 0:
                # Run build commands and print results if verbose is set
                for build_command in benchmark_config["build"]:
                    # Just continue if commmand is empty
                    if build_command == "":
                        continue

                    build_result = subprocess.run(build_command.split(), cwd=current_benchmark_dir,
                                                  stdout=subprocess.DEVNULL if not verbose else None,
                                                  stderr=subprocess.DEVNULL if not verbose else None)

                    # Ensure the building is ok
                    if build_result.stderr is not None or build_result.returncode != 0:
                        build_successful = False
                        break

                if not build_successful:
                    print(f"{print_helpers['error']} Failed building `{auto_tuner_name}`")

            if build_successful and current_benchmark not in ["scan", "sort", "reduction", "stencil2d"]:
                # Run the benchmark command in the benchmark directory
                print(
                    f"{print_helpers['info']} Starting benchmark `{current_benchmark}` for `{os.path.basename(directory)}`")
                run_command = f"{benchmark_config['run']} --size {problem_size} --technique {tuning_technique}"
                run_result = subprocess.run(run_command.split(), cwd=current_benchmark_dir)

                # Check for errors during benchmarking
                if run_result.stderr is not None or run_result.returncode != 0:
                    print(
                        f"{print_helpers['error']} Benchmark `{current_benchmark}` failed for `{os.path.basename(directory)}`")
                else:
                    print(
                        f"{print_helpers['success']} Finished benchmark `{current_benchmark}` for `{os.path.basename(directory)}`")

                    # Parse created JSONs with results and print them
                    if "results" in benchmark_config and isinstance(benchmark_config["results"], list) and len(
                            benchmark_config["results"]) > 0:
                        for results_file_name in benchmark_config["results"]:
                            # Just continue if file name is empty or isn't a JSON file
                            if results_file_name == "" and ".json" not in results_file_name:
                                continue

                            # Read current results file and parse results
                            print(
                                f"{print_helpers['info']} Best parameters from `{results_file_name.split('.json')[0]}`:")
                            parsed_parameters = retrieve_parameter_results(current_benchmark_dir, results_file_name)

                            # Check if any results found
                            if parsed_parameters is None:
                                print(f"{print_helpers['error']} No parameters found in {results_file_name}")
                                continue

                            # Print each parameter line-by-line
                            for parameter, value in parsed_parameters.items():
                                print(f"\t* {parameter}: \033[93m{value}\033[0m")

                    # Copy all JSON and CSV files to benchmark results directory
                    copy_benchmark_result_files(auto_tuner_name, current_benchmark_dir)

    if found_benchmarks:
        print(f"{print_helpers['success']} Finished running all benchmarks")
    else:
        print(f"{print_helpers['error']} Did not find any benchmarks")


if __name__ == "__main__":
    # Setup CLI parser
    parser = argparse.ArgumentParser(description="Benchmark runner")
    parser.add_argument("--benchmark", "-b", type=str, default=None, help="name of the benchmark (e.g.: sort)")
    parser.add_argument("--auto-tuner", "-a", type=str, default=None, help="auto-tuner to benchmark (e.g.: opentuner)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="print stdout and stderr from building of benchmarks")
    parser.add_argument("--size", "-s", type=int, default=1, help="problem size to the benchmark(s) (e.g.: 2)")
    parser.add_argument("--technique", "-t", type=str, default="brute_force",
                        help="tuning technique to use for the benchmark(s) (e.g.: annealing)")
    arguments = parser.parse_args()

    # Run benchmark given inputs
    run_benchmark(benchmark_name=arguments.benchmark, auto_tuner=arguments.auto_tuner, verbose=arguments.verbose,
                  problem_size=arguments.size, tuning_technique=arguments.technique)
