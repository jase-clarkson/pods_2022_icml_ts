import argparse
import numpy as np
import torch
import random
import os

from config_utils import get_config
from experiment_runner import run_exp


def parse_arguments():
    """Gets the name of the config file from the command line and returns it"""
    parser = argparse.ArgumentParser(
        description="Gradient-Based Non-Stationary Forecaster")
    parser.add_argument(
        '-c',
        '--config_file',
        type=str,
        help="Name of config file within the experiment_configs folder",
    )
    args = parser.parse_args()
    return args


def main():
    c_args = parse_arguments()
    config_file = c_args.config_file
    config = get_config(os.path.join('experiment_configs', config_file))

    # set random seeds
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    try:
        run_exp(config)
    except KeyboardInterrupt:
        print('Stopping Experiment')


if __name__ == '__main__':
    main()
