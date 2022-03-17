from numba import cuda

from tuning_examples.common.globals import STRIDE_GLOBAL
from tuning_examples.common.helpers3 import add_parameter, build_config_space

start_path = '../../../../src/programs'


def create_config_definition_spmv(args):
    start, end = 32, cuda.get_current_device().MAX_THREADS_PER_BLOCK
    stride, inclusive = STRIDE_GLOBAL, True

    parameters = [
        add_parameter("BLOCK_SIZE", "integer", value_range=[start, end, stride, inclusive]),
        add_parameter("PRECISION", "enum", value_list=[32, 64]),
        add_parameter("FORMAT", "enum", value_list=[0, 1, 2, 3, 4]),
        add_parameter("UNROLL_LOOP_2", "boolean"),
        add_parameter("TEXTURE_MEMORY", "boolean"),
    ]
    return build_config_space(parameters)
