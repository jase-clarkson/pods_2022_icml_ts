"""

"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from model.torch_models.grad_forecaster import GradForecaster


def compute_squared_loss(predictions, true_labels):
    return pd.Series((predictions - true_labels) ** 2)


class Validation:
    def __init__(self, model_class, X, y, downweight_scheme_fn=None, model_fit_params={}):
        """
        Class used to perform rolling testing
        :param model_class: model class to use
        :param downweight_scheme_fn: function that returns weight scheme from a single argument (weight scheme len)
        """
        self.model_class = model_class
        self.X = X
        self.y = y
        self.T = len(y)
        self.downweight_scheme_fn = downweight_scheme_fn
        self.model_fit_params = model_fit_params

        # attributes that are computed using oss computation methods
        self.model_weights = None
        self.validation_losses = None
        self.test_losses = None

    def rolling_train_valid_test(self, test_start=30, validation_len=10, update_period=10):
        """
        Returns, for each test time point:
         1. sequence of validation set losses
         2. sequence of rolling out of sample losses
        :param test_start: time point on which to start out of sample prediction
        :param validation_len: number of the rolling in-sample points to use for validation
        :param update_period: number of steps until the next model update
        """

        test_losses = {}
        validation_losses = {}
        model_weights = {}

        # checking first validation point has sufficient data
        assert test_start >= 4, 'require test_start >= 4'

        if 's' in self.model_fit_params:
            assert test_start >= self.model_fit_params['s'] + validation_len + 1, ' require test_start => s + validation_len'
        for t in tqdm(range(test_start, self.T, update_period)):
            X_train = self.X[:(t - validation_len)]
            y_train = self.y[:(t - validation_len)]

            X_validation = self.X[(t - validation_len):t]
            y_validation = self.y[(t - validation_len):t]

            X_test = self.X[t:(t + update_period)]
            y_test = self.y[t:(t + update_period)]
            # downweights
            if self.model_class == GradForecaster:
                model = self.model_class(**self.model_fit_params)

                X_tr = torch.tensor(X_train, dtype=torch.float32)
                y_tr = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
                X_val = torch.tensor(X_validation, dtype=torch.float32)
                y_val = torch.tensor(y_validation, dtype=torch.float32).unsqueeze(1)
                X_t = torch.tensor(X_test, dtype=torch.float32)
                y_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)
                validation_losses[t], lower, eta = model.fit(X_tr, y_tr, X_val, y_val)
                lower.update_lower(eta, X_tr, y_tr, X_val, y_val)
                test_losses[t] = pd.Series(lower.calc_val_loss(X_t, y_t, reduction='none'))
                model_weights[t] = eta
            else:
                weights = {}
                if self.downweight_scheme_fn is not None:
                    weights = {'weights': self.downweight_scheme_fn(t - validation_len)}

                model = self.model_class()

                model.fit(X_train, y_train, **weights, **self.model_fit_params)
                validation_predictions = model.predict(X_validation)
                validation_losses[t] = compute_squared_loss(validation_predictions, y_validation).mean()

                # Refit model to train + validation
                weights = {}
                if self.downweight_scheme_fn is not None:
                    weights = {'weights': self.downweight_scheme_fn(t)}

                X_tr_ext = np.r_[X_train, X_validation]
                y_tr_ext = np.r_[y_train, y_validation]
                model = self.model_class()
                model.fit(X_tr_ext, y_tr_ext, **weights, **self.model_fit_params)
                test_predictions = model.predict(X_test)
                test_losses[t] = compute_squared_loss(test_predictions, y_test)

                model_weights[t] = model.theta

        model_weights = pd.Series(model_weights)
        validation_losses = pd.Series(validation_losses)
        test_losses = pd.Series(test_losses)

        self.model_weights = model_weights
        self.validation_losses = validation_losses
        self.test_losses = test_losses

        return validation_losses, test_losses
