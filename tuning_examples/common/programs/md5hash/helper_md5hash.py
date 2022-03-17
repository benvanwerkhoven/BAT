from numba import cuda

from tuning_examples.common.globals import STRIDE_GLOBAL
from tuning_examples.common.helpers3 import add_parameter, build_config_space

start_path = '../../../../src/programs'


def create_config_definition_md5hash(args):
    start, end = 32, cuda.get_current_device().MAX_THREADS_PER_BLOCK
    stride, inclusive = STRIDE_GLOBAL, True

    parameters = [
         add_parameter("BLOCK_SIZE", "integer", value_range=[start, end, stride, inclusive]),
         add_parameter("ROUND_STYLE", "boolean"),
         add_parameter("UNROLL_LOOP_1", "boolean"),
         add_parameter("UNROLL_LOOP_2", "boolean"),
         add_parameter("UNROLL_LOOP_3", "boolean"),
         add_parameter("INLINE_1", "boolean"),
         add_parameter("INLINE_2", "boolean"),
         add_parameter("WORK_PER_THREAD_FACTOR", "enum", value_list=[1, 2, 3, 4, 5])
    ]
    return build_config_space(parameters)
