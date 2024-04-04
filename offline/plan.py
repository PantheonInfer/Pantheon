import shutil

from pymoo.core.problem import Problem, ElementwiseProblem
from pymoo.core.variable import Binary, Real
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival
from pymoo.core.mixed import MixedVariableGA
from pymoo.operators.sampling.rnd import BinaryRandomSampling
from pymoo.operators.crossover.pntx import TwoPointCrossover
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.visualization.scatter import Scatter
from pymoo.optimize import minimize
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse
import json
import os
import importlib
from google.protobuf.text_format import MessageToString
from proto import model_config_pb2


class PlacementPlan(ElementwiseProblem):
    def __init__(self, profiles, setting, **kwargs):
        self.profiles = profiles
        self.num = len(profiles)
        self.max_delta_accuracy = []
        self.orig_accuracy = []
        for key in setting.keys():
            if key == 'max_memory':
                self.max_memory = setting[key]
            else:
                self.max_delta_accuracy.append(setting[key]['max_delta_accuracy'])

        vars = dict()
        for i, profile in enumerate(self.profiles):
            # self.orig_accuracy.append(profile['accuracy'].iloc[-1])
            self.orig_accuracy.append(max(profile['accuracy']))
            on_pareto = pareto(profile['latency'], profile['accuracy'])

            # plt.plot(self.profiles[i]['latency'][on_pareto], self.profiles[i]['accuracy'][on_par  eto], 'ro--')
            # plt.show()

            for j, v in enumerate(on_pareto):
                if True == v and self.orig_accuracy[i] - profile['accuracy'][j] <= self.max_delta_accuracy[i]:
                    vars[f'x{i}:{j}'] = Binary()
        # super().__init__(vars=vars, n_obj=len(self.profiles), n_constr=1 + len(self.profiles), **kwargs)
        super().__init__(vars=vars, n_obj=1, n_constr=1 + len(self.profiles), **kwargs)

    def _evaluate(self, X, out, *args, **kwargs):
        # out['F'] = [-self._cal_sched(X, i) for i in range(self.num)]
        out['F'] = sum([-self._cal_sched(X, i) for i in range(self.num)])
        out['G'] = [sum([self._cal_memory(X,i) for i in range(self.num)]) - self.max_memory] + \
            [max(self.profiles[i]['accuracy']) - self._cal_max_acc(X, i) for i in range(len(self.profiles)) ]

    def _cal_max_acc(self, X, i):
        exits = []
        for k, v in X.items():
            if f'x{i}' in k and True == v:
                idx = int(k.split(':')[-1])
                exits.append(idx)
        if len(exits) == 0:
            return 0
        accuracy = [self.profiles[i]['accuracy'].iloc[j] for j in exits]
        return max(accuracy)

    def _cal_sched(self, X, i):
        no_exit = True
        for k, v in X.items():
            if f'x{i}' in k and True == v:
                no_exit = False
        if no_exit:
            return 0
        T = [self._cal_t(X, i, j) for j in range(len(self.profiles[i]))]
        # return 1 / (max(T) + 0.00)
        return 1 / np.mean(T)

    def _cal_memory(self, X, i):
        exits = []
        for k, v in X.items():
            if f'x{i}' in k and True == v:
                idx = int(k.split(':')[-1])
                exits.append(idx)
        if len(exits) == 0:
            return 0
        last_exit = max(exits)
        mem = sum(self.profiles[i]['block mem'].iloc[:last_exit + 1]) + \
            sum([self.profiles[i]['branch mem'].iloc[idx] for idx in exits])
        return mem

    def _cal_t(self, X, i, j):
        exit_idx = np.inf
        for k, v in X.items():
            if f'x{i}' in k:
                idx = int(k.split(':')[-1])
                if idx >= j and True == v:
                    if exit_idx > idx:
                        exit_idx = idx
        if exit_idx == np.inf:
            return 0
        else:
            return sum(self.profiles[i]['block latency'].iloc[j:exit_idx + 1]) + self.profiles[i]['branch latency'].iloc[exit_idx]

def main(args):
    with open(args.setting, 'r') as f:
        setting = json.load(f)

    models = [key for key in setting.keys() if key != 'max_memory']
    profiles = [pd.read_csv(os.path.join(args.base, m, 'profile.csv')) for m in models]
    for i, profile in enumerate(profiles):
        num_modules = len(profile)
        latency = [sum(profile['block latency'][:i + 1]) + profile['branch latency'][i] for i in range(num_modules)]
        profiles[i]['latency'] = latency


    problem = PlacementPlan(profiles, setting)
    algorithm = MixedVariableGA(pop_size=100, survival=RankAndCrowdingSurvival())
    res = minimize(problem,
                   algorithm,
                   termination=('n_gen', args.ga_gen),
                   seed=1,
                   verbose=True
                   )

    # print("Best solution found: %s" % res.X)
    # print("Function value: %s" % res.F)
    # print("Constraint violation: %s" % res.CV)
    # plot = Scatter()
    # plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
    # plot.add(res.F, facecolor="none", edgecolor="red")
    # plot.show()
    print(res.X)
    if args.save != None:
        build_model_repository(res.X, profiles, models, args)

def pareto(x, y):
    assert len(x) == len(y)
    on_pareto = np.zeros(len(x), dtype=bool)
    for i in range(len(x)):
        on_pareto[i] = np.all(y[x <= x[i]] <= y[i])
    return on_pareto

def build_model_repository(plan, profiles, models, args):
    if os.path.exists(args.save):
        shutil.rmtree(args.save)
    os.mkdir(args.save)
    for model in models:
        os.mkdir(os.path.join(args.save, model))
        os.mkdir(os.path.join(args.save, model, 'model_files'))
    for i, profile in enumerate(profiles):
        exits = []
        for k, v in plan.items():
            if f'x{i}' in k and True == v:
                idx = int(k.split(':')[-1])
                exits.append(idx)
        last_exit = max(exits)
        for j in range(last_exit + 1):
            src = os.path.join(args.base, models[i], 'block_{:02d}.pth'.format(j))
            dst = os.path.join(args.save, models[i], 'model_files', 'block_{:02d}.pth'.format(j))
            shutil.copy(src, dst)
        for j in exits:
            src = os.path.join(args.base, models[i], 'branch_{:02d}.pth'.format(j))
            dst = os.path.join(args.save, models[i], 'model_files', 'branch_{:02d}.pth'.format(j))
            shutil.copy(src, dst)
        model_config = model_config_pb2.ModelConfig()
        model_config.name = models[i]
        configs = importlib.import_module(f'dnn.{models[i]}.configs')
        dims = [1] + configs.INPUT_SHAPE
        for dim in dims:
            model_config.dims.append(dim)

        for j in range(last_exit + 1):
            block_profile = model_config_pb2.ModelBlockProfile()
            block_profile.id = j
            block_profile.latency = int(profile['block latency'][j] * 1000)
            model_config.block_profile.append(block_profile)
            if j in exits:
                exit_profile = model_config_pb2.ModelBlockProfile()
                exit_profile.id = j
                exit_profile.latency = int(profile['branch latency'][j] * 1000)
                exit_profile.accuracy = profile['accuracy'][j]
                model_config.exit_profile.append(exit_profile)
        with open(os.path.join(args.save, models[i], 'config.pbtxt'), 'w') as f:
            f.write(MessageToString(model_config))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=str, default='../experiments/weights')
    parser.add_argument('--setting', type=str, default='../experiments/settings/deploy/demo.json')
    parser.add_argument('--save', type=str, default='../experiments/model_repositories/demo')
    parser.add_argument('--ga_gen', type=int, default=20)
    args = parser.parse_args()
    main(args)