from numba import cuda

from tuning_examples.common.globals import STRIDE_GLOBAL
from tuning_examples.common.helpers3 import add_parameter, build_config_space

start_path = '../../../../src/programs'


def create_config_definition_bfs(args):
    sizes = [1000, 10000, 100000, 1000000, 10000000]
    numVerts = sizes[args.size - 1]

    start, end = 32, min(numVerts, cuda.get_current_device().MAX_THREADS_PER_BLOCK)
    stride, inclusive = STRIDE_GLOBAL, True

    parameters = [
        add_parameter("BLOCK_SIZE", "integer", value_range=[start, end, stride, inclusive]),
        add_parameter("CHUNK_FACTOR", "enum", value_list=[1, 2, 4, 8]),
        add_parameter("TEXTURE_MEMORY_EA1", "enum", value_list=[0, 1, 2]),
        add_parameter("TEXTURE_MEMORY_EAA", "enum", value_list=[0, 1, 2]),
    ]
    return build_config_space(parameters)
