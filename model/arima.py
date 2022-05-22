"""
Implement ARIMA(p, d, q) model. Used as a competitor method to DBF in Kuznetsov et al.

p, d, q are hyperparameters to optimise over on the validation set.

"""

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from model.model_utils import Model


class Arima(Model):
    def __init__(self, theta=None):
        """
        Implements a ARIMA(p, d, q) model
        """
        super(Arima, self).__init__()
        self.model = None
        # theta stores Arima result parameters - not the same as the ar coefficients in other Models
        self.theta = theta

    def fit(self, X, y, p, d, q, trend='n'):
        """
        :param X: not used (should correspond to lagged values of y)
        :param y: univariate time series
        :param p: order of autoregressive component
        :param d: order of difference
        :param q: order of moving average component

        """
        self.model = ARIMA(endog=y, order=(p, d, q), trend=trend, enforce_stationarity=False,
                           enforce_invertibility=False)
        self.model = self.model.fit(method='statespace', cov_type='none')
        self.theta = self.model.params


    def predict(self, X):
        """
        Return predictions for exog n x p matrix X
        :return: n x 1 vector of predictions
        """
        if self.model is None:
            raise ValueError('self.model is None')
        else:
            # UserWarning('assumes that new time series values embedded in the first column of X extend '
            #             'the training set y')
            # recovering the univariate time series from X's first column (corresponding the lag-1 values of the
            # time series y

            y_extension = X[:, 0]
            if len(y_extension) == 1:
                prediction = self.model.forecast()
            else:
                y_extension = y_extension[1:]

                prediction = list(self.model.forecast())

                for y_val in y_extension:
                    self.model = self.model.extend([y_val])
                    prediction.extend(self.model.forecast())

                prediction = np.array(prediction)

            return prediction


if __name__ == '__main__':
    # testing model
    from model.model_utils import generate_features_labels
    import pandas as pd
    y = pd.Series(np.arange(100))
    X, y = generate_features_labels(y, model_type='univariate', n_lags=2)
    X_train, y_train = X[:80], y[:80]
    X_valid, y_valid = X[80:], y[80:]
    p = 1
    d = 0
    q = 0
    trend = 'n'
    model = Arima()
    model.fit(X_train, y_train, p, d, q, trend)
    print(model.predict(X_valid))
