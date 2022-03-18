import json
from itertools import product
from random import randint

from tuning_examples.common.benchmark_helpers import compile_benchmark, run_benchmark
from tuning_examples.common.helpers2 import result_builder
from tuning_examples.min_tuner.progressbar import Progressbar


class MinTuner:

    def __init__(self, args):
        with open(args.path, 'r') as f:
            self.oadc = json.load(f)

    @staticmethod
    def cartesian_space_builder(cfg_def):
        params = []
        for param in cfg_def['parameters']:
            v = param['values']
            if v.get('value_list', None):
                v_list = list(v['value_list'])
                params.append(v_list)
            if v.get('value_range', None):
                v_range = list(range(int(v['value_range']['start']),
                                     int(v['value_range']['end']) + int(v['value_range']['inclusive'] == True),
                                     int(v['value_range']['stride'])))
                params.append(v_range)
        return list(product(*params))

    @staticmethod
    def list_to_dict(cfg_def, cfg_list):
        d = []
        for i, param in enumerate(cfg_def['parameters']):
            r = dict()
            r['name'] = param['name']
            r['value'] = cfg_list[i]
            d.append(r)
        return d

    def brute_force(self, cfg_def, i):
        cartesian_space = self.cartesian_space_builder(cfg_def)
        return self.list_to_dict(cfg_def, cartesian_space[i])

    def random_search(self, cfg_def):
        cartesian_space = self.cartesian_space_builder(cfg_def)
        return self.list_to_dict(
            cfg_def, cartesian_space[randint(0, cfg_def['cardinality'])])

    def pick_config(self, cfg_def, search_settings, i):
        name = search_settings['search_technique']['name']
        if name == "brute_force":
            return self.brute_force(cfg_def, i)
        elif name == "random_search":
            return self.random_search(cfg_def)

    def run(self):
        all_results = []
        meta = self.oadc['metadata']
        r = range(meta['search_settings']['budget']['steps'])
        prog = Progressbar(len(r))
        prog.algorithm(meta['benchmark_suite']['benchmark_name'])
        for i in r:
            cfg = self.pick_config(meta['configuration_space'], meta['search_settings'], i)
            compile_result = compile_benchmark(meta, cfg)
            run_result = run_benchmark(meta)
            result = result_builder(compile_result, run_result, cfg)
            all_results.append(result)
            prog.update_progress(i)

        return all_results
