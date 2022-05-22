"""
script to run grid experiment
"""

import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pickle
import numpy as np
import pandas as pd
import multiprocessing as mp
from datetime import datetime
from sklearn.model_selection import ParameterGrid
from grid_experiment_utils import Experiment, run_experiment_wrapper
from functools import partial
from downweighting_schemes import span_to_alpha
from real_data_constants import train_valid_test_params_dict, s_dict, lookback_list_dict

from model.torch_models.weighting_schemes import *  # For scheme grid construction

def run_exp(config):
    # checking if SLURM multiprocessing
    try:
        ncpu = os.environ["SLURM_JOB_CPUS_PER_NODE"]
        ncpu = sum(int(num) for num in ncpu.split(','))
        print('number of SLURM CPUS: ', ncpu)
    except KeyError:
        ncpu = config.experiment.ncpu

    if hasattr(config.dataset, 'real_data'):
        real_data = config.dataset.real_data
    else:
        real_data = False
    if real_data:
        data_generation_param_grid = setup_real_data_problem(config)
    else:
        data_generation_param_grid = setup_synthetic_data_problem(config)

    feature_label_param_grid = [config.dataset.feature_labels]

    # In the yaml files, any parameters shared by multiple models are indicated by a string naming
    # the variable. These must be set up here and put into the shared_params dict.
    shared_params = {}
    for k, v in config.model_shared_params.items():
        if k == 'reg_weight_list':
            bounds = v.log_reg_weight_bounds
            shared_params['reg_weight_list'] = np.r_[0, np.logspace(bounds[0], bounds[1], v.n_values)]
        elif k == 'lookback_list':
            if real_data:  # If real data use lookback list according to freq
                frequency = config.experiment.frequency
                lookback_list = lookback_list_dict[frequency]
            else:  # Else use the values given
                lookback_list = list(map(np.int, np.linspace(v.lb, config.dataset.data_generation.T[0], v.n_values)))
            # Compute parameters for grid search optimised downweighted forecasters.
            shared_params['lookback_list'] = lookback_list
            shared_params['alpha_list'] = span_to_alpha(lookback_list)
        elif k == 's_dict':  # Set the value of s for dbf forecaster on real datasets to match freq.
            assert real_data
            shared_params['s_dict'] = [s_dict[config.experiment.frequency]]

    # Loop through the model parameters, replacing any strings with
    # the appropriate value from shared_params.
    model_param_grid = []
    for model in config.baselines:
        model_dict = {}
        for k, v in model.items():
            if type(v) == str:
                model_dict[k] = shared_params[v]
            else:
                model_dict[k] = v
        model_param_grid.append(model_dict)
    # Create the grid for GradForecaster - multiple sections in arg file
    # so we do a 'grid of grids' structure
    config.model_args.scheme_args = setup_schemes(config.model_args.scheme_args)
    gf_grid = {k: ParameterGrid(v) for k, v in config.model_args.items()}
    gf_param_grid = list(ParameterGrid(gf_grid))

    data_generation_param_grid = list(ParameterGrid(data_generation_param_grid))
    feature_label_param_grid = list(ParameterGrid(feature_label_param_grid))
    model_param_grid = gf_param_grid + list(ParameterGrid(model_param_grid))

    # number of experiment reps
    if real_data:
        experiment_list = [
            Experiment([data_generation_param_grid[ticker_num]],
                       feature_label_param_grid,
                       model_param_grid,
                       config.save_dir,
                       ticker_num) for
            ticker_num, _ in enumerate(data_generation_param_grid)
        ]
        num_reps = len(data_generation_param_grid)

    else:
        num_reps = config.experiment.num_rep_per_cpu * ncpu
        experiment_list = [Experiment(data_generation_param_grid.copy(),
                                      feature_label_param_grid.copy(),
                                      model_param_grid.copy(),
                                      config.save_dir,
                                      seed) for seed in range(num_reps)]

    run_experiment_wrapper_partial = partial(run_experiment_wrapper,
                                             test_start=config.experiment.test_start,
                                             validation_len=config.experiment.validation_len,
                                             update_period=config.experiment.update_period)

    start_time = datetime.now()
    print(f'Starting experiment now at {start_time}')
    if ncpu > 1:
        with mp.Pool(ncpu, maxtasksperchild=4) as pool:
            pool.map(run_experiment_wrapper_partial, experiment_list)
    else:
        list(map(run_experiment_wrapper_partial, experiment_list))
    end_time = datetime.now()
    print(f'Experiment ended at {end_time}')
    print(f'Total Runtime {end_time - start_time}')

    # Load the CSVs

    results_frame = [pd.read_pickle(os.path.join(config.save_dir, f'{seed}.pkl')) for seed in range(num_reps)]
    results_frame = {rep: df for rep, df in enumerate(results_frame)}

    results_frame = pd.concat(results_frame, axis=1, names=['rep', 'results'])

    results_frame.to_pickle(os.path.join(config.save_dir, 'results.pkl'))
    with open('data/experiments/last_run.txt', 'w') as f:
        f.write(config.save_dir)


def setup_real_data_problem(config):
    data_set = config.dataset.data_generation.data_set
    problem = config.dataset.data_generation.problem
    # Load the tickers for the dataset
    with open('./data/real_data/' + data_set + '/' + 'ticker_list' + '.pkl', 'rb') as f:
        ticker_list = pickle.load(f)
    data_generation_param_grid = [
        dict(data_generation_fn_name=['generate_real_data'],
             problem=[problem],
             ticker=[ticker],
             data_set=[data_set]) for ticker in ticker_list
    ]
    # Set the train/test/val periods for the experiment
    train_valid_test_params = train_valid_test_params_dict[config.experiment.frequency]
    config.experiment.test_start = train_valid_test_params['test_start']
    config.experiment.validation_len = train_valid_test_params['validation_len']
    config.experiment.update_period = train_valid_test_params['update_period']
    return data_generation_param_grid

def setup_synthetic_data_problem(config):
    problem = config.dataset.data_generation.pop('problem')
    config.dataset.data_generation.data_generation_fn_name = ['generate_' + problem]
    data_generation_param_grid = [config.dataset.data_generation]
    return data_generation_param_grid

def setup_schemes(scheme_args):
    grid = []
    for scheme_name in scheme_args['name']:
        scheme = eval(scheme_name)
        # Fixed number parameters - don't grid over num_basis_fn (deprecated for ICML sub).
        if hasattr(scheme, 'eta_dim'):
            grid.append({'name': [scheme_name]})
        else:
            grid.append({'name': [scheme_name], 'n_basis_fn': scheme_args['n_basis_fn']})
    return grid
