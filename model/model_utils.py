"""
Model util functions

"""

import numpy as np
import pandas as pd


def generate_features_labels(data, model_type, **kwargs):
    """
    Generate the feature matrix and vector of labels for AR model fit problem.
    Note: If n_lags > 1, no intercept term is included.
    """

    if model_type == 'univariate':

        assert 'n_lags' in kwargs, 'n_lags not passed as key word argument'

        y = data.values

        n_lags = kwargs['n_lags']

        if n_lags == 0:
            X = np.ones(shape=(data.shape[0], 1))
        else:
            X = {str(lag): data.shift(lag).fillna(0) for lag in range(1, n_lags + 1)}
            X = pd.DataFrame(X)
            X = X.values

    elif model_type == 'factor':

        feature_names = ['mktrf', 'smb', 'hml']
        ticker = data.columns[[name not in feature_names for name in data.columns]]
        ticker = ticker[0]
        y = data.loc[:, ticker]
        y = y.values

        # add intercept term
        X = data.loc[:, feature_names]
        X = X.values
        X = np.c_[np.ones(len(y)), X]

    elif model_type == 'statespace':
        # assuming generate_state_space returns a tuple of this form
        y, X, beta = data
        
    else:
        raise 'model_type not detected'

    return X, y


#%% Model class

class Model:
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        """
        Return predictions for exog n x p matrix X
        :return: n x 1 vector of predictions
        """
        pass

    def grad_predictions(self, X):
        """
        Return gradient wrt model parameters of predictions at data X
        :param X: exog n x p matrix at which to evaluate predictions gradient
        :return: n x p matrix of gradients of predictions
        """
        pass

    def second_grad_predictions(self, X):
        """
        Return tensor of second derivatives of predictions wrt model parameters evaluated at X
        :param X: exog n x p matrix at which to evaluate predictions gradient
        :return: n x (p x p) tensor of second derivative matrices
        """
        pass
