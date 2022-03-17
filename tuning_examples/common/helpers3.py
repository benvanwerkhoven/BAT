import subprocess
import time

from numba import cuda


def add_parameter(name, type, value_list=[], value_range=[]):
    if value_list:
        values = {"value_list": value_list}
    elif value_range:
        values = {
            "value_range": {
                "start": value_range[0],
                "end": value_range[1],
                "stride": value_range[2],
                "inclusive": value_range[3]
            }
        }
    elif type == "boolean":
        values = {"value_list": [0, 1]}
    else:
        values = {}
    return {
        "name": name,
        "type": type,
        "values": values
    }


def calculate_meta(parameters):
    cardinality = 1
    for param in parameters:
        if param['type'] in ('enum', 'boolean'):
            cardinality *= max(1, len(param['values']['value_list']))
        else:
            s = param['values']['value_range']
            cardinality *= max(1, len(range(int(s['start']), int(s['end'])+int(s['inclusive']), int(s['stride']))))
    return cardinality, len(parameters)


def build_config_space(parameters):
    cardinality, dimensionality = calculate_meta(parameters)
    return {
        "parameters": parameters,
        "cardinality": cardinality,
        "dimensionality": dimensionality,
    }


def call_program(cmd):
    start_time = time.time()
    run_error = subprocess.run([cmd],
                               shell=True, capture_output=True).stderr.decode()
    end_time = time.time()
    duration = end_time - start_time
    correctness = 0.0 if run_error else 1.0
    result = {
        'error': run_error,
        'time': duration,
        'correctness': correctness
    }
    # print(result)
    return result


def run_cmd_builder(search_settings, launch_args, benchmark_name):
    # args = search_settings['parallelization']
    program_command = f'./{benchmark_name} -s ' + str(launch_args['size'])
    if launch_args['parallel']:
        # Select number below max connected GPUs
        chosen_gpu_number = min(launch_args['gpunum'], len(cuda.gpus))

        devices = ','.join([str(i) for i in range(-1, chosen_gpu_number)])
        run_cmd = f'mpirun -np {chosen_gpu_number} --allow-run-as-root {program_command} -d {devices}'
    else:
        run_cmd = program_command
    return run_cmd
