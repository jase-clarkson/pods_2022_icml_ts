import numpy as np
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import functorch
import matplotlib.pyplot as plt
from functorch import grad, vjp
from functools import partial
from tqdm import tqdm

from model.torch_models.weighting_schemes import *
from model.torch_models.lower_forecasters import *



def compute_ihp(v, f, theta, alpha, i=3):
    # This function computes an approximation of the vector * H^-1 product, where
    # H^-1 is the inverse of the hessian of f. We compute this blockwise for each parameter.
    # This function is an implementation of the psuedocode given in https://arxiv.org/pdf/1911.02590.pdf.
    theta = list(theta)
    p = [vec.detach().clone() for vec in v]
    v = list(vec.clone() for vec in v)
    vjp_fun = vjp(f, theta)[1]
    for j in range(i):
        vjps = list(vjp_fun(v)[0])  # This returns a tuple, with the vhp for each parameter block
        for k, param in enumerate(vjps):
            # Loop through each parameter block and iterate
            v[k] -= alpha * vjps[k]
            p[k] -= v[k]
    return p


def compute_hypergradient(model, eta, lr, scheme, X_tr, y_tr, X_val, y_val, reg_upper):
    func_model, theta = functorch.make_functional(model)
    # dtr_dtheta, dtr_dw = grad(tr_loss, argnums=(0, 1))(params, weights)
    val_loss = partial(compute_validation_loss, data=X_val, targets=y_val, model=func_model, lambda_=reg_upper)
    dval_dtheta, dval_deta = grad(val_loss, argnums=(0, 1))(theta, eta)

    # Using the notation of the Duvenaud paper.
    v1 = dval_dtheta  # each part of v1 is different block of parameters in the gradient vector.

    # We can compute the vector-matrix product for each part of the vector and concatenate them at the end.

    ## We also need a partial that fixes the weights, so that we can take Hessian w.r.t theta
    tr_loss_theta = partial(compute_weighted_loss,
                            eta=eta, data=X_tr, targets=y_tr, model=func_model, scheme=scheme)
    grad_tr_theta = grad(tr_loss_theta)
    # v2 is dval/dtheta * H^-1
    v2 = compute_ihp(v1, grad_tr_theta, theta, lr)  # Returns v*H for each block of params in v
    # v2 = v1
    ####### Fix all args except theta, eta
    tr_loss = partial(compute_weighted_loss,
                      data=X_tr, targets=y_tr, model=func_model, scheme=scheme)
    d_tr_dtheta = lambda eta_: grad(tr_loss, argnums=0)(theta, eta_)
    v3 = vjp(d_tr_dtheta, eta)[1](tuple(v2))[0]
    # If using identity, times by -1
    return v3 + dval_deta


def compute_weighted_loss(theta, eta, data, targets, model, scheme):
    preds = model(theta, data)
    sq_err = (preds - targets)**2
    weights = scheme.compute_scheme(data.shape[0], eta).unsqueeze(1)
    weighted_errors = sq_err * weights
    return torch.mean(weighted_errors)


def compute_validation_loss(params, eta, data, targets, model, lambda_=0.005):
    preds = model(params, data)
    return torch.mean((preds - targets)**2) + lambda_ * torch.linalg.vector_norm(eta, ord=1)


class TsDataset(Dataset):
    def __init__(self, X, y):
        assert X.shape[0] == y.shape[0]
        self.X = X
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_eta_inits(n_params, n_restarts, scales=(3, 8)):
    exponent = np.random.uniform(scales[0], scales[1], size=(n_restarts, n_params))
    inits = np.power([10.0], -1 * exponent)
    inits[0] *= 1e-32
    return inits


def get_inits(scheme, n_params, n_restarts):
    return get_eta_inits(n_params, n_restarts)


class GradForecaster:
    def __init__(self, lower_args,
                 scheme_args, upper_args,
                 training_args, opt_args):
        self.lower = eval(lower_args['name'])
        self.lower_args = lower_args.copy()
        self.lower_args.pop('name')
        if 'lr' in lower_args:
            self.lower_lr = lower_args['lr']
        else:  # Lower is fit exactly.
            self.lower_lr = 1
        self.n_epochs = training_args['n_epochs']
        self.batch_size = training_args['batch_size']
        self.clip_grad = training_args['clip_grad']
        self.n_restarts = training_args['n_restarts']

        self.scheme = eval(scheme_args['name'])
        if hasattr(self.scheme, 'eta_dim'):
            self.n_params = self.scheme.eta_dim
        else:
            self.n_params = scheme_args['n_basis_fn']

        self.upper_args = upper_args

        opt = opt_args['opt']
        # Store opt information
        if opt == 'Adam':
            self.opt = optim.Adam
        elif opt == 'SGD':
            self.opt = optim.SGD
        self.opt_args = opt_args.copy()
        self.opt_args.pop('opt')
        return

    def fit(self, X_tr, y_tr, X_val, y_val, tag=None):
        eta_inits = get_inits(self.scheme, self.n_params, self.n_restarts)
        dataset = TsDataset(X_val, y_val)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        validation_losses = np.array([np.inf])
        eta_list = []
        grad_list = []
        # theta_list = []
        best_model = None
        best_loss = None
        best_eta = None
        for eta_ in eta_inits:
            eta = torch.tensor(eta_)
            eta = torch.nn.Parameter(eta)
            opt = self.opt([eta], **self.opt_args)
            for i in range(self.n_epochs):
                for X_val_batch, y_val_batch in loader:
                    lower = self.lower(input_dim=X_tr.shape[1], scheme=self.scheme, **self.lower_args)
                    training_loss = lower.fit(eta, X_tr, y_tr)
                    # Compute total validation loss.
                    val_loss = lower.calc_val_loss(X_val, y_val)
                    if (val_loss < validation_losses).all():
                        best_model = lower
                        best_loss = val_loss
                        best_eta = eta.detach().numpy()
                    validation_losses = np.append(validation_losses, [val_loss])
                    eta_list.append(eta.detach().clone().numpy())
                    opt.zero_grad()
                    # Evaluate validation loss at current weights

                    eta_grad = compute_hypergradient(lower, eta, self.lower_lr, self.scheme,
                                                     X_tr, y_tr, X_val_batch, y_val_batch, **self.upper_args)
                    grad_list.append(torch.linalg.vector_norm(eta_grad).detach().numpy())
                    eta.grad = eta_grad
                    nn.utils.clip_grad_norm_([eta], self.clip_grad)
                    opt.step()
        return best_loss, best_model, best_eta

    def plot_results(self, training_losses,
                     validation_losses, eta_list, grad_list, train_len, save_path=None):
        fig, axs = plt.subplots(1, 5, figsize=(12, 4))
        axs[0].plot(training_losses)
        axs[0].set_title('Training Losses')
        axs[1].plot(pd.Series(validation_losses).rolling(100).mean())
        axs[1].set_title('Val Loss')
        axs[2].plot(grad_list)
        axs[2].set_title('Eta Grad Norm')
        min_iter = np.argmin(validation_losses)
        min_eta = eta_list[min_iter]
        print(f'Min Iteration: {min_iter} || Min Eta: {min_eta}')
        axs[3].plot([np.linalg.norm(eta_val) for eta_val in eta_list])
        axs[3].set_title('Eta')
        final_weights = self.scheme.compute_scheme(train_len, torch.tensor(min_eta)).detach().numpy()
        axs[4].plot(final_weights)
        axs[4].set_title('final_weights')
        plt.tight_layout()
        if save_path is None:
            plt.show()
        else: # Save
            plt.savefig(save_path)




