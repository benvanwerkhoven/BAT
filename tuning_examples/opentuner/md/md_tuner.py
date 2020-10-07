#!/usr/bin/env python

import opentuner
from opentuner import ConfigurationManipulator
from opentuner import IntegerParameter
from opentuner import EnumParameter
from opentuner.search.manipulator import BooleanParameter
from opentuner import MeasurementInterface
from opentuner import Result
from numba import cuda
import math
import json

start_path = '../../../src/programs'

class MDTuner(MeasurementInterface):
    all_results = []

    def manipulator(self):
        """
        Define the search space by creating a
        ConfigurationManipulator
        """

        gpu = cuda.get_current_device()
        max_block_size = gpu.MAX_THREADS_PER_BLOCK
        manipulator = ConfigurationManipulator()
        # Using block size less than `gpu.MAX_THREADS_PER_BLOCK`
        manipulator.add_parameter(IntegerParameter('BLOCK_SIZE', 1, max_block_size))
        manipulator.add_parameter(EnumParameter('PRECISION', [32, 64]))
        manipulator.add_parameter(IntegerParameter('TEXTURE_MEMORY', 0, 1))
        manipulator.add_parameter(IntegerParameter('WORK_PER_THREAD', 1, 5))

        return manipulator

    def run(self, desired_result, input, limit):
        """
        Compile and run a given configuration then
        return performance
        """
        args = argparser.parse_args()

        cfg = desired_result.configuration.data
        compute_capability = cuda.get_current_device().compute_capability
        cc = str(compute_capability[0]) + str(compute_capability[1])

        make_program = f'nvcc -gencode=arch=compute_{cc},code=sm_{cc} -I {start_path}/cuda-common -I {start_path}/common -g -O2 -c {start_path}/md/md.cu'
        make_program += ' -D{0}={1}'.format('BLOCK_SIZE', cfg['BLOCK_SIZE'])
        make_program += ' -D{0}={1}'.format('PRECISION', cfg['PRECISION'])
        make_program += ' -D{0}={1}'.format('TEXTURE_MEMORY', cfg['TEXTURE_MEMORY'])
        make_program += ' -D{0}={1} \n'.format('WORK_PER_THREAD', cfg['WORK_PER_THREAD'])

        if args.parallel:
            make_paralell_start = f'mpicxx -I {0}/common/ -I {0}/cuda-common/ -I /usr/local/cuda/include -DPARALLEL -I {start_path}/mpi-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_paralell_end = f'mpicxx -L {start_path}/cuda-common -L {start_path}/common -o md main.o md.o -lSHOCCommon "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib/stubs" "-L/usr/local/cuda/bin/../targets/x86_64-linux/lib" -lcudadevrt -lcudart_static -lrt -lpthread -ldl -lrt -lrt'
            compile_cmd = make_paralell_start + make_program + make_paralell_end
        else:
            make_serial_start = f'nvcc -I {start_path}/common/ -I {start_path}/cuda-common/ -g -O2 -c -o main.o {start_path}/cuda-common/main.cpp \n'
            make_serial_end = f'nvcc -L {start_path}/cuda-common -L {start_path}/common -o md main.o md.o -lSHOCCommon'
            compile_cmd = make_serial_start + make_program + make_serial_end

        compile_result = self.call_program(compile_cmd)
        assert compile_result['returncode'] == 0

        program_command = './md -s ' + str(args.size)
        if args.parallel:
            # Select number below max connected GPUs
            chosen_gpu_number = min(args.gpu_num, len(cuda.gpus))

            devices = ','.join([str(i) for i in range(0, chosen_gpu_number)])
            run_cmd = f'mpirun -np {chosen_gpu_number} --allow-run-as-root {program_command} -d {devices}'
        else:
            run_cmd = program_command

        run_result = self.call_program(run_cmd)

        # Check that error code and error output is ok
        assert run_result['stderr'] == b''
        assert run_result['returncode'] == 0

        result = {'parameters': cfg, 'time': run_result['time']}
        self.all_results.append(result)
        return Result(time=run_result['time'])

    def save_final_config(self, configuration):
        """called at the end of tuning"""
        print("Optimal parameter values written to results.json:", configuration.data)
        with open('all-results.json', 'w') as f:
            json.dump(self.all_results, f, indent=4)
        
        # Update configuration with problem size and tuning technique
        configuration.data["PROBLEM_SIZE"] = argparser.parse_args().size
        configuration.data["TUNING_TECHNIQUE"] = argparser.parse_args().technique
        
        self.manipulator().save_to_file(configuration.data, 'results.json')


if __name__ == '__main__':
    argparser = opentuner.default_argparser()
    argparser.add_argument('--size', type=int, default=1, help='problem size of the program (1-4)')
    argparser.add_argument('--gpu-num', type=int, default=1, help='number of GPUs')
    argparser.add_argument('--parallel', action="store_true", help='run on multiple GPUs')
    MDTuner.main(argparser.parse_args())
