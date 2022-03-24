from tuning_examples.common.helpers3 import call_program, run_cmd_builder

from tuning_examples.common.programs.bfs.helper_bfs import create_config_definition_bfs
from tuning_examples.common.programs.md.helper_md import create_config_definition_md
from tuning_examples.common.programs.md5hash.helper_md5hash import create_config_definition_md5hash
from tuning_examples.common.programs.reduction.helper_reduction import create_config_definition_reduction
from tuning_examples.common.programs.scan.helper_scan import create_config_definition_scan
from tuning_examples.common.programs.sort.helper_sort import create_config_definition_sort
from tuning_examples.common.programs.spmv.helper_spmv import create_config_definition_spmv
from tuning_examples.common.programs.stencil2d.helper_stencil2d import create_config_definition_stencil2d
from tuning_examples.common.programs.triad.helper_triad import create_config_definition_triad


def compile_benchmark(meta, cfg, gpu):
    parallel = False
    benchmark_name = meta['benchmark_suite']['benchmark_name']
    compile_cmd = f'make -j8 {"parallel" if parallel else "default"} PARAMETERS="'
    # print(cfg)
    for param in cfg:
        compile_cmd += f' -D{param["name"]}={param["value"]}'
    compile_cmd += f'" BIN_NAME="{benchmark_name}-{gpu}"\n'

    return call_program(compile_cmd)


def run_benchmark(meta, gpu):
    benchmark_name = meta['benchmark_suite']['benchmark_name']
    return call_program(run_cmd_builder(
        meta['search_settings'],
        meta['benchmark_suite']['args'],
        benchmark_name, gpu))


def_dict = {
    "BFS": create_config_definition_bfs,
    "MD": create_config_definition_md,
    "MD5HASH": create_config_definition_md5hash,
    "REDUCTION": create_config_definition_reduction,
    "SCAN": create_config_definition_scan,
    "SORT": create_config_definition_sort,
    "SPMV": create_config_definition_spmv,
    "STENCIL2D": create_config_definition_stencil2d,
    "TRIAD": create_config_definition_triad,
}


def create_config_definition(args):
    return def_dict[args.benchmark.upper()](args)
