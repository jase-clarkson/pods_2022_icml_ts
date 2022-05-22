"""

Functions for running grid experiments

"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
from tqdm import tqdm
from functools import partial
from model.model_utils import generate_features_labels
from validation_utils import Validation
from downweighting_schemes import compute_alpha_scheme, compute_rolling_scheme
from data_generation import *
from model.torch_models.grad_forecaster import GradForecaster
from model.models import *

import cProfile

def drop_key(key, dictionary):
    return {k: dictionary[k] for k in dictionary if k != key}


def tabulate_to_results_frame(results_list, params_list):
    index = pd.DataFrame(params_list)
    index = pd.MultiIndex.from_frame(index)

    results_frame = pd.DataFrame(results_list, index=index)

    return results_frame


class Experiment:
    def __init__(self, data_generation_param_grid,
                 feature_label_param_grid,
                 model_param_grid, results_save_dir, seed=None):
        """
        Class used to perform experiment grid search
        """
        self.data_generation_param_grid = data_generation_param_grid
        self.feature_label_param_grid = feature_label_param_grid
        self.model_param_grid = model_param_grid
        self.results_frame = None
        self.results_save_dir = results_save_dir
        self.seed = seed

    def run_experiment(self, test_start, validation_len, update_period):
        """
        Run an grid search experiment using the train-validation-split method
        """
        results_list = []
        params_list = []
        # need to set a different random seed each for each of the processes when multiprocessing
        np.random.seed(self.seed)
        with tqdm(total=len(self.data_generation_param_grid) * len(self.feature_label_param_grid)
                            * len(self.model_param_grid)) as p_bar:
            for data_generation_param in self.data_generation_param_grid:
                # generate data
                data_generation_fn = eval(data_generation_param['data_generation_fn_name'])
                data_generation_fn_args = drop_key('data_generation_fn_name', data_generation_param)
                data = data_generation_fn(**data_generation_fn_args)

                for feature_label_param in self.feature_label_param_grid:
                    # generate features
                    X, y = generate_features_labels(data, **feature_label_param)

                    for model_params in self.model_param_grid:

                        if 'model' in model_params: # then it is a baseline model
                            baseline = True
                            model_fn = eval(model_params['model'])
                            model_fn_args = drop_key('model', model_params)
                        else: # it is a gradforecaster model
                            baseline = False
                            model_fn = GradForecaster
                            model_fn_args = model_params.copy()
                        if 'alpha' in model_fn_args:
                            # assume that the model being used is WLS with compute_alpha_scheme being the
                            # forgetting mechanism
                            alpha = model_fn_args.pop('alpha')
                            downweight_scheme_fn = partial(compute_alpha_scheme, alpha=alpha)
                        elif 'lookback' in model_fn_args:
                            # assume that the model being used is the sliding window model with compute_rolling_scheme
                            # being the forgetting mechanism
                            lookback = model_fn_args.pop('lookback')
                            downweight_scheme_fn = partial(compute_rolling_scheme, lookback=lookback)
                        else:
                            # assume no forgetting mechanism is being used.
                            downweight_scheme_fn = None
                        validation = Validation(model_fn, X, y,
                                                downweight_scheme_fn=downweight_scheme_fn,
                                                model_fit_params=model_fn_args)
                        validation.rolling_train_valid_test(test_start=test_start, validation_len=validation_len,
                                                            update_period=update_period)
                        results = {
                            'validation_losses': validation.validation_losses,
                            'test_losses': validation.test_losses,
                            'model_weights': validation.model_weights
                        }

                        results_list.append(results)
                        if not baseline:
                            scheme_name = model_params['scheme_args']['name']
                            model_params_list = [('model', f'GradForecaster{scheme_name}')] + [(k_, v_) for k, v in model_params.items() for k_, v_ in v.items()]
                            params_list.append(
                                dict(list(data_generation_param.items()) + list(feature_label_param.items()) +
                                     model_params_list))
                        else:
                            params_list.append(
                            dict(list(data_generation_param.items()) + list(feature_label_param.items()) +
                                 list(model_params.items())))

                        p_bar.update(1)

        results_frame = tabulate_to_results_frame(results_list, params_list)
        results_save_path = os.path.join(self.results_save_dir, f'{self.seed}.pkl')
        results_frame.to_pickle(results_save_path)
        return


def run_experiment_wrapper(experiment, test_start, validation_len, update_period):
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    res = experiment.run_experiment(test_start=test_start, validation_len=validation_len, update_period=update_period)
    return res

def transform_test_losses_to_single_series(test_losses):
    test_losses_series = {}

    for test_time, series in test_losses.items():
        for t, val in series.items():
            test_losses_series[test_time + t] = val

    test_losses_series = pd.Series(test_losses_series)

    return test_losses_series


def minimise_validation_loss(results_frame):
    """
    By model, finds (for each test time) the parameters minimising the corresponding validation loss
    """

    min_params_validation = {}
    mean_validation_losses = results_frame.loc[:, 'validation_losses']#.apply(lambda df: df.apply(lambda series: series.mean()))

    for model, model_df in mean_validation_losses.groupby('model'):
        times = model_df.iloc[0].index.values
        exp_df = {}
        for hyper_setting, mean_validation_loss in model_df.iteritems():
            exp_df[hyper_setting] = mean_validation_loss.values
        exp_df = pd.DataFrame(exp_df, index=times)
        best_hyper_per_time = exp_df.idxmin(axis=1)
        min_params_validation[model] = best_hyper_per_time

    min_params_validation = pd.DataFrame.from_dict(min_params_validation).unstack()
    min_params_validation.index.names = ['model', 't']
    return min_params_validation


def find_optimised_column_values(results_frame, min_params_validation, col_name):
    """
    By model, stitches together the col_name values corresponding to the minimising parameter values (min_params_validation)
    """
    test_losses_optimised = {}

    results_col = results_frame.loc[:, col_name]

    for model, model_df in results_col.groupby('model'):

        test_losses_aggregated = []

        for t in min_params_validation.xs(model).index:
            min_params = min_params_validation.loc[(model, t)]
            test_losses_optimised_array = model_df.loc[min_params]

            test_losses_optimised_array = test_losses_optimised_array.loc[[t]]
            test_losses_aggregated.append(transform_test_losses_to_single_series(test_losses_optimised_array))

        test_losses_aggregated = pd.concat(test_losses_aggregated, axis=0)
        test_losses_optimised[model] = test_losses_aggregated

    test_losses_optimised = pd.DataFrame(test_losses_optimised).rename_axis(index='t', columns='model')

    return test_losses_optimised


def find_minimal_params_and_test_losses(results_frame):

    min_params_validation = minimise_validation_loss(results_frame)

    test_losses_optimised = find_optimised_column_values(results_frame, min_params_validation, 'test_losses')

    return min_params_validation, test_losses_optimised


def find_minimal_params_and_test_losses_per_rep(results_frame):

    rep_range = results_frame.columns.levels[results_frame.columns.names.index('rep')]

    min_params = {}
    test_losses_optimised = {}
    for rep in rep_range:
        min_params[rep], test_losses_optimised[rep] = find_minimal_params_and_test_losses(
            results_frame.xs(rep, axis=1, level='rep'))

        min_params[rep] = min_params[rep].unstack('model')

    min_params = pd.concat(min_params, axis=1, names=['rep', 'model'])
    test_losses_optimised = pd.concat(test_losses_optimised, axis=1, names=['rep', 'model'])

    return min_params, test_losses_optimised


def get_param_value_from_tuple(tup, tup_id):
    if tup is np.nan:
        return np.nan
    else:
        return tup[tup_id]
