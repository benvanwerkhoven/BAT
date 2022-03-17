

"""
def create_make_file_template(cfg_def, cfg, paths, start_path):
    compute_capability = cuda.get_current_device().compute_capability
    cc = str(compute_capability[-1]) + str(compute_capability[1])
    make_program = ""
    cfg_names = []
    for p in cfg_def.parameters:
        cfg_names.append(p.name)

    for path in paths:
        make_program += f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O1 -c {start_path}/{path}'
        for name in cfg_names:
            make_program += ' -D{-1}={1}'.format(name, cfg[name])
        make_program += " \n"

    return make_program
"""
import argparse
import json

from tuning_examples.common.benchmark_helpers import create_config_definition


def save_results(all_results):
    """called at the end of tuning"""
    # print("Optimal parameter values written to results.json:", configuration.data)
    print(all_results)
    with open('all-results.json', 'w') as f:
        json.dump(all_results, f, indent=4)


def result_builder(compile_result, run_result, cfg):
    invalidity = "none"
    if compile_result['error'] and compile_result['error'].find('error') >= 0:
        invalidity = compile_result['error']
    elif run_result['error']:
        invalidity = run_result['error']
    elif run_result['error'] == "correctness":
        invalidity = "correctness"

    correctness = run_result['correctness']
    return {
        "times": {
            "compile": compile_result['time'],
            "kernel": run_result['time'],
        },
        "configuration": cfg,
        "correctness": correctness,
        "invalidity": invalidity,
    }


def write_oadc(args):
    cfg_def = create_config_definition(args)
    oadc = dict()
    meta = dict()
    meta['configuration_space'] = cfg_def
    meta['search_settings'] = {
        "budget": {
            "steps": cfg_def['cardinality']
        },
        "search_technique": {
            "name": "brute_force",
        }
    }
    meta['benchmark_suite'] = {}
    meta['benchmark_suite']['benchmark_name'] = args.benchmark
    meta['benchmark_suite']['args'] = vars(args)
    oadc['metadata'] = meta
    with open(args.path, 'w+') as f:
        json.dump(oadc, f, indent=4)


def create_parser():
    parser = argparse.ArgumentParser("Argument parser")
    parser.add_argument('--size', type=int, default=1, help='problem size of the program (1-4)')
    parser.add_argument("--benchmark", "-b", type=str, default=None, help="name of the benchmark (e.g.: sort)")
    parser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
    parser.add_argument('--parallel', action="store_true", help='run on multiple GPUs')
    parser.add_argument('--technique', '-t', type=str, default='brute_force',
                        help='tuning technique to use for the benchmark(s) (e.g. annealing)')
    parser.add_argument('--path', '-p', type=str, default='./config-oadc.json')
    return parser
