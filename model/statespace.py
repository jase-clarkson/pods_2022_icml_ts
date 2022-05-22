import numpy as np
import pandas as pd
import statsmodels.api as sm
from model.model_utils import Model
from data_generation import generate_state_space_rw

#%% Implementing a RW-coefficient state space model


class TVRegression(sm.tsa.statespace.MLEModel):
    def __init__(self, y, x):
        # renaming x
        exog = x

        # the states of the kalman filter/model correspond to time-varying coefficients
        k_states = 1 if len(x.shape) < 2 else x.shape[1]

        super(TVRegression, self).__init__(
            endog=y, exog=exog, k_states=k_states, initialization="diffuse"
        )

        # Since the design matrix is time-varying, it must be
        # shaped k_endog x k_states x nobs
        # Notice that exog.T is shaped k_states x nobs, so we
        # just need to add a new first axis with shape 1
        self.ssm["design"] = exog.T[np.newaxis, :, :]  # shaped 1 x k_states x nobs
        self.ssm["selection"] = np.eye(self.k_states)
        self.ssm["transition"] = np.eye(self.k_states)

        # Which parameters need to be positive? --> var.e and all vars for the coefficients corresponding to exog
        # variables x
        self.positive_parameters = slice(0, self.k_states + 1)

        # need to set k_exog to k_states for the forecast method to work
        self.k_exog = self.k_states

    @property
    def param_names(self):
        return ["var.e"] + ['var.x' + str(x_ind) + '.coeff' for x_ind in range(1, self.k_states + 1)]

    @property
    def start_params(self):
        """
        Defines the starting values for the parameters
        The linear regression gives us reasonable starting values for the constant
        d and the variance of the epsilon error
        """
        # exog = sm.add_constant(self.exog)
        res = sm.OLS(self.endog, self.exog).fit()
        params = np.r_[res.scale, [0.001] * self.k_states]
        return params

    def transform_params(self, unconstrained):
        """
        We constraint the last three parameters
        ('var.e', 'var.x.coeff', 'var.w.coeff') to be positive,
        because they are variances
        """
        constrained = unconstrained.copy()
        constrained[self.positive_parameters] = (
            constrained[self.positive_parameters] ** 2
        )
        return constrained

    def untransform_params(self, constrained):
        """
        Need to unstransform all the parameters you transformed
        in the `transform_params` function
        """
        unconstrained = constrained.copy()
        unconstrained[self.positive_parameters] = (
            unconstrained[self.positive_parameters] ** 0.5
        )
        return unconstrained

    def update(self, params, **kwargs):
        params = super(TVRegression, self).update(params, **kwargs)

        # self["obs_intercept", 0, 0] = params[0]
        self["obs_cov", 0, 0] = params[1]
        self["state_cov"] = np.diag(params[1:(self.k_states + 1)])


class StateSpace(Model):
    def __init__(self, theta=None):
        """
        Implements a coefficient RW-drift state space model
        """
        super(StateSpace, self).__init__()
        self.theta = theta
        self.model = None

    def fit(self, X, y):
        self.model = TVRegression(y, X)
        # Note that the covariance matrix of the error increment in the statespace formula is not set to
        # be diagonal in this formulation
        self.model = self.model.fit(maxiter=100, disp=0, cov_type='none', method='lbfgs')
        self.theta = self.model.filtered_state.T[-1]

    def predict(self, X):
        if self.model is None:
            raise ValueError('self.model is None')
        else:
            prediction = np.dot(X, self.model.filtered_state.T[-1])
            return prediction

