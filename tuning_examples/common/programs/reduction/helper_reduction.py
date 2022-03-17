from numba import cuda

from tuning_examples.common.globals import STRIDE_GLOBAL
from tuning_examples.common.helpers3 import add_parameter, build_config_space

start_path = '../../../../src/programs'


def create_config_definition_reduction(args):
    start, end = 32, cuda.get_current_device().MAX_THREADS_PER_BLOCK
    stride, inclusive = STRIDE_GLOBAL, True

    parameters = [
        add_parameter("BLOCK_SIZE", "integer", value_range=[start, end, stride, inclusive]),
        add_parameter("GRID_SIZE", "enum", value_list=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        add_parameter("PRECISION", "enum", value_list=[32, 64]),
        add_parameter("COMPILER_OPTIMIZATION_HOST", "enum", value_list=[0, 1, 2, 3]),
        add_parameter("COMPILER_OPTIMIZATION_DEVICE", "enum", value_list=[0, 1, 2, 3]),
        add_parameter("USE_FAST_MATH", "boolean"),
        add_parameter("MAX_REGISTERS", "enum", value_list=[-1, 20, 40, 60, 80, 100, 120]),
        add_parameter("GPUS", "enum", value_list=[1+i for i in range(args.gpu_num)]),
        add_parameter("LOOP_UNROOL_REDUCE_1", "boolean"),
        add_parameter("LOOP_UNROOL_REDUCE_2", "boolean"),
        add_parameter("TEXTURE_MEMORY", "boolean"),
    ]
    return build_config_space(parameters)
