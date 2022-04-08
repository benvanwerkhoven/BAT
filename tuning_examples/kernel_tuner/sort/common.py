import cupy as cp
import json
from numpyencoder import NumpyEncoder

def store_BAT_results(benchmark_name, tuning_results, input_problem_size, strategy, tune_params):

    # Save the results as a JSON file
    with open(f"{benchmark_name}-results.json", 'w') as f:
        json.dump(tuning_results, f, indent=4, cls=NumpyEncoder)

    # Get the best configuration
    best_parameter_config = min(tuning_results[0], key=lambda x: x['time'])
    best_parameters = dict()

    # Filter out parameters from results
    for k, v in best_parameter_config.items():
        if k not in tune_params:
            continue

        best_parameters[k] = v

    # Add problem size and tuning technique to results
    best_parameters["PROBLEM_SIZE"] = input_problem_size
    best_parameters["TUNING_TECHNIQUE"] = strategy

    # Save the best results as a JSON file
    with open(f"best-{benchmark_name}-results.json", 'w') as f:
        json.dump(best_parameters, f, indent=4, cls=NumpyEncoder)

def get_device_info(device):
    """ Get device info using cupy """
    result = dict()

    cupy_info = str(cp._cupyx.get_runtime_info()).split("\n")[:-1]
    info_dict = {s.split(":")[0].strip():s.split(":")[1].strip() for s in cupy_info}
    result["device_name"] = info_dict[f'Device {device} Name']

    with cp.cuda.Device(0) as dev:

        result['max_threads'] = dev.attributes['MaxThreadsPerBlock']
        result['max_shared_memory'] = dev.attributes['MaxSharedMemoryPerBlock']

    return result
