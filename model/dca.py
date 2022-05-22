# coding: utf-8

"""
Taken from https://github.com/121onto/trust-region-dca.
A python implementation of a DC algorithm for the trust-region sub-problem.
Copyright (C) 2015, 121onto.  All rights reserved.
Notes
-----
Solves the following problem:
.. math::
    min_{x} x^TAx + b^Tx
    s.t. \| x \| \le r
References
----------
Pham Dinh Tao and Le Thi Hoai An,
"A DC Optimization Algorithm for Solving the Trust-Region Subproblem,"
SIAM Journal on Optimization 8, no. 2 (1998): 476–505.
"""

###########################################################################
## imports

import warnings
import numpy as np
import scipy.sparse.linalg as sparse_linalg

###########################################################################
## local solver

class TrustRegionDCA(object):
    """Solves the trust-region sub-problem using a DC algorithm.
    """
    def __init__(self, A, b, r, n=None,
                 stopping_tol=10e-6, irlm_tol=0.1, verbose=False):

        self.A = A
        self.b = b
        self.r = r
        self.n = n if n is not None else len(A)

        self.irlm_tol = irlm_tol
        self.stopping_tol = stopping_tol

        self.lambda_star = None
        self.lambda_n = self._irlm(tol=irlm_tol, which='LA', return_eigenvectors=False)
        # eigsh does not converge with tol=0
        self.lambda_1, self.u = self._irlm(tol=10e-1, which='SA', return_eigenvectors=True)

        self.xk_0 = None
        self.xk_1 = self._get_initial_xk(self.n, r)

        self.cycle = False
        self.x_local_cycle_check = None
        self.x_global_cycle_check = None

        self.rho = self._get_initial_rho()
        self.cut = self.rho * self.r
        self.A_alt = np.eye(self.n) * self.rho - self.A

        self.total_iters = 0
        self.verbose = verbose
        if verbose:
            print('lambda_1: {}'.format(self.lambda_1))
            print('lambda_n: {}'.format(self.lambda_n))
            print('lambda_star: {}'.format(self.lambda_star))
            print('rho: {}'.format(self.rho))
            print('irlm_tol: {}'.format(self.irlm_tol))
            print('stopping_tol: {}'.format(self.stopping_tol))


    def _irlm(self, tol=None, which='LA', return_eigenvectors=False):
        # assert(which in ['LM','SM'])
        tol = self.irlm_tol if tol is None else tol
        if return_eigenvectors:
            vals, vecs  = sparse_linalg.eigsh(
                self.A,
                k=1,
                which=which,
                tol=tol,
                return_eigenvectors=return_eigenvectors
            )
            return vals[0], vecs[:,0]

        vals  = sparse_linalg.eigsh(
            self.A,
            k=1,
            which=which,
            tol=tol,
            return_eigenvectors=return_eigenvectors
        )
        return vals[0]


    def _get_initial_rho(self):
        """TODO: consider alternatives"""
        return 0.5 * self.lambda_n


    def _get_initial_xk(self, n, r, eps=10e-4):
        """TODO: consider alternatives, perhaps random???"""
        assert(n>0)
        x0 = np.empty(n)
        x0.fill(r/np.sqrt(n))
        return x0


    def reset_xk(self):
        """Reset xk when local DCA fails to find a global solution"""
        # TODO: test this, or at least reinspect a few times to ensure accuracy
        if self.lambda_star is None:
            self.update_lambda_star()

        x_star = self.xk_1
        bx_star = np.dot(self.b, x_star)

        if bx_star > 0:
            self.xk_1 *= -1.0
            return None

        u = self.u
        norm_x_star = np.linalg.norm(x_star)
        ux_star = u.dot(x_star)
        norm_u_sqrd = np.linalg.norm(u) ** 2.
        gamma = None

        if np.isclose(norm_x_star, self.r):
            while np.isclose(ux_star, 0.0):
                bu = np.dot(self.b, u)
                # uA_lu < 0 when x is not a global optimum, see equation (22)
                uA_lu = (self.lambda_1 + self.lambda_star) * norm_u_sqrd
                tau = None

                if np.isclose(bx_star, 0):
                    if bu <= 0:
                        # any tau < 0 will do
                        tau = -1.
                    else:
                        # any 0 > tau > u'(A + lambda_star I)u / b'u will do
                        tau = 0.5 * uA_lu / bu
                else:
                    # tau the smallest root of -b'x* tau ** 2 - 2b'u tau + u'(A + lambda_star I)u
                    tau = (bu + np.sqrt(bu ** 2. + bx_star * uA_lu)) / bx_star

                u = u + tau * x_star
                ux_star = np.dot(u, x_star)
                norm_u_sqrd = np.linalg.norm(u) ** 2.

        if np.isclose(norm_x_star, self.r):
            gamma = -2. * ux_star / norm_u_sqrd

        else:
            sqrt_delta = np.sqrt(
                ux_star ** 2. - norm_u_sqrd * (norm_x_star ** 2. - self.r ** 2.)
            )
            if ux_star < 0:
                gamma = (sqrt_delta - ux_star) / norm_u_sqrd
            else:
                gamma = -1.0 * (sqrt_delta + ux_star) / norm_u_sqrd

        self.xk_1 = self.xk_1 + gamma * u
        return None


    def update_xk(self):
        """Update step for local DCA"""
        self.xk_0 = self.xk_1
        self.xk_1 = self.A_alt.dot(self.xk_1) - self.b
        norm = np.linalg.norm(self.xk_1)
        if norm <= self.cut:
            self.xk_1 /= self.rho
        else:
            self.xk_1 *= (self.r / norm)

    def update_lambda_star(self):
         self.lambda_star = -1 * np.dot(
             self.xk_1,
             np.dot(self.A, self.xk_1) + self.b
         ) / self.r ** 2.


    def solve_local_dca(self):
        self.update_xk()
        dist = np.linalg.norm(self.xk_0 - self.xk_1)
        while dist >= self.stopping_tol:
            self.x_local_cycle_check = self.xk_0
            self.update_xk()

            if np.allclose(self.x_local_cycle_check, self.xk_1):
                self.cycle = True
                break
            else:
                dist = np.linalg.norm(self.xk_0 - self.xk_1)

    def solve_global_dca(self):
        stop = False
        while not stop:
            if self.lambda_star is not None:
                self.reset_xk()

            self.x_global_cycle_check = self.xk_1
            self.solve_local_dca()
            self.update_lambda_star()

            if np.allclose(self.x_global_cycle_check, self.xk_1):
                self.cycle = True

            if self.cycle:
                break

            stop = (self.lambda_star + self.lambda_1 >= 0) or self.cycle

            self.total_iters += 1
            if self.total_iters > 10:
                print('max iters reached')
                break

        if self.cycle:
            warnings.warn(
                'Cycle detected, the solution may not be unique'
                ' or another error occurred.'
            )

        return self.xk_1

###########################################################################
## tests

def test_trdca_convex(prng, n=100, r=10, verbose=False):
    # Generate A
    A = np.diag([(i + 1) / n for i in range(n)])
    b = np.zeros(n)

    # initialize solver
    trdca = TrustRegionDCA(
        A = A,
        b = b,
        r = r,
        n = n,
        verbose=verbose
    )
    x_star_dca = trdca.solve_global_dca()

    x_star_truth = np.zeros(n)
    x_star_truth[-1] = 10.


    def obj(x):
        return np.abs(0.5 * x.dot(A).dot(x) + b.dot(x))

    # solution is not unique
    assert((
        np.allclose(x_star_dca, x_star_truth, atol=10e-6) or
        np.allclose(x_star_dca, x_star_truth * -1., atol=10e-6)
        ))


def test_trdca_nonconvex(prng, n=100, r=10, verbose=False):
    """TODO: work in progress"""
    # Generate A (skip for now)
    U = np.eye(n)
    for i in range(3):
        w = prng.uniform(-1,1,n);
        w[w == -1] = 0.
        Q = np.eye(n) - 2 * np.outer(w,w) / np.linalg.norm(w) ** 2.
        U = U.dot(Q)

    d = prng.uniform(-5, 5, n)
    d[d == -5] = 0.
    A = U.dot(np.diag(d)).dot(U.T)

    # Generate b
    g = prng.uniform(-1, 1, n)
    g[g == -1] = 0.
    b = U.dot(g)

    # initialize solver
    trdca = TrustRegionDCA(
        A = A,
        b = b,
        r = r,
        n = n,
        verbose=verbose
    )

    def fun(x):
        return 0.5 * x.dot(A).dot(x) + b.dot(x)

    def jac(x):
        return (0.5 * x.dot(A + A.T) + b.T)

    x_star_dca = trdca.solve_global_dca()
    x_star_slsqp = optimize.minimize(
        fun,
        prng.uniform(-1,1,n),
        method='SLSQP',
        jac=jac,
        constraints=dict(
            type='ineq',
            fun = lambda x: r ** 2. - x.dot(x),
            jac = lambda x: -2. * x.T
        )
    ).x

    assert(np.allclose(fun(x_star_dca), fun(x_star_slsqp), atol=10e-3))

###########################################################################
## main

if __name__ == '__main__':
    import scipy.optimize as optimize
    try:
        from config import SEED
    except:
        SEED = 1237777

    prng = np.random.RandomState(SEED)
    test_trdca_convex(prng, n=1000)
    # test_trdca_nonconvex(prng, n=1000)

    print('\n\n--> All tests ran without error.\n\n')