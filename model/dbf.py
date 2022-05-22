"""
Using the method of https://arxiv.org/pdf/1803.05814.pdf, in particular using the global solution to the
difference of convex functions.
"""
import os
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import pandas as pd
import cvxpy as cp
import multiprocessing as mp
from scipy.optimize import minimize, NonlinearConstraint
from multiprocessing import Pool
from functools import partial
import matplotlib.pyplot as plt
from tqdm import tqdm
from numba import jit
from model.model_utils import Model
from model.dca import TrustRegionDCA


class DiscrepProblem:
    # This is a wrapper object for the functions required for the discrepancy computation.
    # These include the objective function, the constraint and the Jacobian and Hessian functions
    # for both of these.
    def __init__(self, A, b, c, mult=1, lambda_=1):
        assert(mult == 1 or mult == -1)
        # Objective w'Aw + b'w + c
        self.obj = lambda w: mult*(np.dot(w, np.dot(A, w)) + np.dot(w, b) + c)
        # d(obj)/dw = 2*Aw + b
        self.obj_j = lambda w: mult*(2*np.dot(A, w) + b)
        # d(jac)/dw = 2*A
        self.obj_h = lambda w: mult*2*A
        # Constraint ||w||^2 = w'w <= Lambda^2
        constr = lambda w: np.dot(w, w)
        # d(constr)/dw = 2*w
        constr_j = lambda w: 2*w
        # d(cons_jac)/dw_idw_j = 2 if i=j, 0 otherwise
        constr_h = lambda w, v: v[0] * 2 * np.diag(np.ones(len(w)))
        cons_pos = NonlinearConstraint(constr, lb=-np.inf, ub=lambda_**2, jac=constr_j, hess=constr_h)
        constr_n = lambda w: -np.dot(w, w)
        # d(constr)/dw = 2*w
        constr_j_n = lambda w: -2*w
        # d(cons_jac)/dw_idw_j = 2 if i=j, 0 otherwise
        constr_h_n = lambda w, v: -v[0] * 2 * np.diag(np.ones(len(w)))
        cons_neg = NonlinearConstraint(constr_n, lb=-(lambda_**2), ub=np.inf, jac=constr_j_n, hess=constr_h_n)
        self.cons = [cons_pos, cons_neg]

    def solve(self, w0):
        return minimize(self.obj, w0, method='trust-constr', jac=self.obj_j, hess=self.obj_h,
                         constraints=self.cons)


def compute_discrepancies(X, y, s, lambda_=1, parallel=False):
    T = X.shape[0]
    t_list = list(t for t in range(T))
    discrep_partial = partial(compute_discrepancy, X=X, y=y, s=s, lambda_=lambda_)
    if parallel:
        with Pool(mp.cpu_count() - 1) as p:
            discrepancies = p.map(discrep_partial, t_list)
    else:
        discrepancies = map(discrep_partial, t_list)
    return list(discrepancies)


def compute_discrepancy(t, X, y, s, lambda_=1):
    T = X.shape[0]
    N = X.shape[1]
    # P = diag(1/s, ..., 1/s, -1), where 1/s is repeated s times
    P = np.diag((s * [1/s]) + [-1])
    # X = (x_1, ... , x_n, x_t)
    X_ = np.vstack((X[T-s:, :], X[t, :]))
    # A = X'PX
    A = np.matmul(X_.T, np.matmul(P, X_))
    # b = -2((sum y_i * p_i * x_i) - y_t*x_t)
    b = np.sum([-2*y[T-s+i]*P[i,i]*X[T-s+i, :] for i in range(s)] + [2*y[t]*X[t, :]], axis=0)
    # c = sum{p_k(y_k)^2} - (y_t)^2 - p_k is 0 further than s, and just 1/s otherwise
    c = np.mean(y[T-s:]**2) - y[t]**2
    # Maximise |w'Aw + b'w + c| subject to ||w||^2 <= Lambda^2
    # == Minimise -|...| st ...
    # Note as we are using a minimize function the 'positive' version corresponds to minimizing the negative version
    # and vice-versa
    def obj(x):
        return np.dot(x, np.dot(A, x)) + np.dot(b, x) + c

    # Min -w'Aw - b'w - c
    trdca = TrustRegionDCA(
        A = -1 * A,
        b = -1 * b,
        r = lambda_
    )
    res_p = trdca.solve_global_dca()
    p_val = -1 * obj(res_p)
    # Min w'Aw + bw + c
    trdca = TrustRegionDCA(
        A = A,
        b = b,
        r = lambda_
    )
    res_n = trdca.solve_global_dca()
    n_val = obj(res_n)
    # w0 = np.ones(N)
    # pos = DiscrepProblem(A, b, c, mult=-1, lambda_=lambda_)
    # res_p = pos.solve(w0)
    # # Maximise -w'Aw - b'w - c s.t. ...
    # neg = DiscrepProblem(A, b, c, mult=1, lambda_=lambda_)
    # res_n = neg.solve(w0)
    # return abs(min(res_p.fun, res_n.fun))
    return abs(min(p_val, n_val))


class QOptimiser:
    def __init__(self, X, y, d, v, lambda_2):
        self.w = cp.Parameter(X.shape[1])
        self.q = cp.Variable(X.shape[0])
        self.z = cp.Variable()
        # Note: The problem is written in the following way so the objective is affine in all parameters and variables.
        # This enables us to use DPP, which caches the solution path to the problem for faster subsequent solves.
        self.err = (X @ self.w) - y
        self.weighted_disc = self.q @ d
        self.reg_q = lambda_2 * 1/10000 * cp.norm(10000 * (self.q - v), p=2)
        # self.obj = cp.Minimize(self.q @ (self.err @ self.z + d) + self.weighted_disc + self.reg_w + self.reg_q)
        self.obj = cp.Minimize(self.q @ (self.err**2 + d) + self.reg_q)

        # self.cons = [self.q <= 1, self.q >= 0, self.z == cp.diag(self.q) @ self.err]
        self.cons = [self.q <= 1, self.q >= 0]

        self.prob = cp.Problem(self.obj, self.cons)

    def solve(self, w):
        self.w.value = w
        self.prob.solve()
        return self.q.value


class WOptimiser:
    def __init__(self, X, y, lambda_1, lambda_=1):
        self.q = cp.Parameter(X.shape[0], nonneg=True)
        self.w = cp.Variable(X.shape[1])
        self.obj = cp.Minimize((self.q @ (X @ self.w - y)**2) + (lambda_1 * cp.norm(self.w)**2))
        self.cons = [cp.norm(self.w, p=2) <= lambda_]
        self.prob = cp.Problem(self.obj, self.cons)

    def solve(self, q):
        try:
            self.q.value = q
        except ValueError:
            self.q.value = np.abs(q)
        self.prob.solve()
        return self.w.value


def fit_db_forecaster(X, y, lambda_1, lambda_2, s,
                      n_iter, lambda_, parallel_discrep=False, plot_discrep=False): # n_iter=5, lambda_=1
    T = X.shape[0]
    N = X.shape[1]
    # Start with fixed q, optimise in w (this is a QP)
    d = compute_discrepancies(X, y, s, lambda_, parallel=parallel_discrep)
    # d = np.array(1000 * [0] + 1000 * [1] + 1000 * [0])
    if plot_discrep:
        plt.plot(d)
        plt.title('Raw Discrepancies')
        plt.show()
        pd.DataFrame(d).rolling(window=30).mean().plot()
        plt.title('Discrepancies after a 30-Lag MA Filter')
        plt.show()
        print('--- Discrepancy Statistics ---')
        print('Mean: ', np.mean(d))
        print('Var: ', np.var(d))
        print('Max: ', np.max(d))
        print('Min: ', np.min(d))
        print('------------------------------')
    # Choose uniform weights for regularisation prior v
    v = np.ones(T) / T
    # Initial value of q taken to be all ones
    q = np.ones(T)
    w = np.ones(N) / N
    q_opt = QOptimiser(X, y, d, v, lambda_2)
    w_opt = WOptimiser(X, y, lambda_1, lambda_)
    for i in range(n_iter):
        q = q_opt.solve(w)
        w = w_opt.solve(q)
    return w


#%% Model class implementation of Kuznetsov's discrepency-based forecaster in the linear case with square loss
class DBFLinear(Model):
    def __init__(self, theta=None):
        super(DBFLinear, self).__init__()
        self.theta = theta

    def fit(self, X, y, lambda_1, lambda_2, s, n_iter, lambda_):
        self.theta = fit_db_forecaster(X=X, y=y,
                                       lambda_1=lambda_1, lambda_2=lambda_2,
                                       s=s, n_iter=n_iter, lambda_=lambda_)

    def predict(self, X):
        if self.theta is None:
            raise ValueError('self.theta is None in DBFLinear')
        else:
            return np.dot(X, self.theta)
