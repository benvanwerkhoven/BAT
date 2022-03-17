from numba import cuda

from tuning_examples.common.globals import STRIDE_GLOBAL
from tuning_examples.common.helpers3 import add_parameter, build_config_space

start_path = '../../../../src/programs'


def create_config_definition_sort(args):
    start, end = 32, cuda.get_current_device().MAX_THREADS_PER_BLOCK
    stride, inclusive = STRIDE_GLOBAL, True

    parameters = [
        add_parameter("LOOP_UNROOL_LSB", "boolean"),
        add_parameter("LOOP_UNROOL_LOCAL_MEMORY", "boolean"),
        add_parameter("SCAN_DATA_SIZE", "enum", value_list=[2, 4, 8]),
        add_parameter("SORT_DATA_SIZE", "enum", value_list=[2, 4, 8]),
        add_parameter("SCAN_BLOCK_SIZE", "integer", value_range=[start, end, stride, inclusive]),
        add_parameter("SORT_BLOCK_SIZE", "integer", value_range=[start, end, stride, inclusive]),
        add_parameter("INLINE_LSB", "boolean"),
        add_parameter("INLINE_SCAN", "boolean"),
        add_parameter("INLINE_LOCAL_MEMORY", "boolean"),
    ]
    return build_config_space(parameters)
