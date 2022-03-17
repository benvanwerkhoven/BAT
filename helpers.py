import os
import json
import shutil
import time

project_dir = os.path.dirname(os.path.abspath(__file__))
benchmark_dir = os.path.join(project_dir, "tuning_examples")
benchmark_start_time = str(int(time.time()))
results_dir = os.path.join(project_dir, "results", benchmark_start_time)

print_helpers = {
    "info": "\033[93m[i]\033[0m",
    "success": "\033[92m[✓]\033[0m",
    "error": "\033[91m[✕]\033[0m"
}


# Return the sub directories of the tuning examples directory
def get_subdirectories(directory):
    # If directory is not project dir check if it is a directory
    if directory != project_dir:
        if not os.path.isdir(directory):
            raise Exception('Input directory does not exists!')

    return [f.path for f in os.scandir(directory) if f.is_dir()]


def retrieve_benchmark_config(benchmark_dir):
    config_file = os.path.join(benchmark_dir, "config.json")

    # Check if the benchmark config file exists in the directory
    if not os.path.isfile(config_file):
        return None

    # Parse the config JSON file
    with open(config_file, 'r') as f:
        config_data = json.load(f)

    # It is required for the config file to contain the run command
    if not "run" in config_data or config_data["run"] == "":
        return None

    return config_data


def retrieve_parameter_results(benchmark_dir, results_file_name):
    results_file = os.path.join(benchmark_dir, results_file_name)

    # Check if the parameter results file exists in the directory
    if not os.path.isfile(results_file):
        return None

    # Parse the results JSON file
    with open(results_file, 'r') as f:
        parameter_results = json.load(f)

    # Check if there are no parameters in the results file
    if len(parameter_results.items()) == 0:
        return None

    return parameter_results


def copy_benchmark_result_files(auto_tuner_name, benchmark_dir):
    # Copy all JSON and CSV files to results directory for this benchmark
    current_results_dir = os.path.join(results_dir, auto_tuner_name, os.path.basename(benchmark_dir))

    # Create the results directory
    os.makedirs(current_results_dir, exist_ok=True)

    current_json_results = [f for f in os.listdir(benchmark_dir) if
                            os.path.isfile(os.path.join(benchmark_dir, f)) and f.endswith(
                                ".json") and f != "config.json"]
    current_csv_results = [f for f in os.listdir(benchmark_dir) if
                           os.path.isfile(os.path.join(benchmark_dir, f)) and f.endswith(".csv")]

    for file in current_json_results + current_csv_results:
        shutil.copy2(os.path.join(benchmark_dir, os.path.basename(file)), current_results_dir)

    print(f"{print_helpers['success']} Copied benchmark results to: {current_results_dir}")


