import numpy as np
import pandas as pd
import statsmodels.api as sm
from model.model_utils import Model

#%% Stationary linear model


class StationaryLinear(Model):
    def __init__(self, theta=None):
        """
        :param theta: linear model parameter vector
        """
        super(StationaryLinear, self).__init__()
        self.theta = theta
        self.model = None

    def fit(self, X, y, reg_weight):
        # model attribute is from WLS statsmodel
        # intercept not included by default
        # lambda_ gives the regularisation weight
        self.model = sm.regression.linear_model.OLS(endog=y, exog=X)
        self.model = self.model.fit_regularized(alpha=reg_weight, L1_wt=0)
        self.theta = self.model.params

    def predict(self, X):

        if self.model is None:
            raise ValueError('self.model is None')
        else:
            return self.model.predict(X)

    def grad_predictions(self, X):

        return X

    def second_grad_predictions(self, X):

        n = X.shape[0]
        p = X.shape[1]

        return np.zeros((n, p, p))


#%% Linear model using user-specified downweighting scheme

class WLSLinear(Model):
    def __init__(self, theta=None):
        '''
        :param theta: linear model parameter vector
        '''
        super(WLSLinear, self).__init__()
        self.theta = theta
        self.model = None

    def fit(self, X, y, weights, reg_weight):
        # model attribute is from WLS statsmodel
        # intercept not included by default
        # lambda_ gives the regularisation weight
        self.model = sm.regression.linear_model.WLS(endog=y, exog=X, weights=weights)
        self.model = self.model.fit_regularized(alpha=reg_weight, L1_wt=0)
        self.theta = self.model.params

    def predict(self, X):

        if self.model is None:
            raise ValueError('self.model is None')
        else:
            return self.model.predict(X)

    def grad_predictions(self, X):

        return X

    def second_grad_predictions(self, X):

        n = X.shape[0]
        p = X.shape[1]

        return np.zeros((n, p, p))


#%% Linear model using user-specified sliding window scheme

class SlidingLinear(WLSLinear):
    """
    SlidingLinear class is implemented using WLSLinear
    """
    pass


