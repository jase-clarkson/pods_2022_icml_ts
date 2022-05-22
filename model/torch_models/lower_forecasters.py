import torch
import torch.nn as nn
import torch.nn.functional as F


class WLSGD(torch.nn.Module):
    def __init__(self, input_dim, scheme, reg_lower, bias=False):
        super(WLSGD, self).__init__()
        self.linear1 = nn.Linear(input_dim, 1, bias=bias)
        self.scheme = scheme
        self.lambda_ = reg_lower
        self.bias = bias

    def forward(self, x):
        pred = self.linear1(x)
        return pred

    def solve_normal_eqs(self, X, weights, y):
        # Setup the normal equations
        XTW = X.t() * weights
        lhs = XTW @ X
        if self.lambda_ > 0:  # Add ridge penalty if non-zero
            lhs += torch.eye(lhs.shape[0]) * self.lambda_

        rhs = XTW @ y
        # Solve normal equations for theta.
        theta = torch.linalg.lstsq(lhs, rhs).solution
        return theta

    def fit(self, eta, X, y):
        self.train()
        t = X.shape[0]
        # Don't want grad w.r.t eta for solving lower.
        weights = self.scheme.compute_scheme(t, eta).detach()

        # Solve normal equations for theta.
        theta = self.solve_normal_eqs(X, weights, y)
        # Set the value of theta (for compatability with bi-level codebase)
        self.set_parameters(theta)
        # Make predictions for training data and compute the less
        preds = self.forward(X)
        # Below actually just computes the squared errors.
        loss = F.mse_loss(preds, y, reduction='none')
        # Compute the in-sample training error
        best_loss = torch.mean(loss * weights.unsqueeze(1)).item()
        return best_loss

    def update_lower(self, eta, X_tr, y_tr, X_val, y_val):
        with torch.no_grad():
            weights = self.scheme.compute_scheme(X_tr.shape[0], torch.tensor(eta))
            final_weight = weights[-1]
            val_len = X_val.shape[0]
            weights = torch.concat([weights, torch.ones([val_len]) * final_weight])
            X_new = torch.concat([X_tr, X_val])
            y_new = torch.concat([y_tr, y_val])
            theta = self.solve_normal_eqs(X_new, weights, y_new)
            self.set_parameters(theta)

    def set_parameters(self, theta):
        if self.bias:
            self.linear1.bias.data = theta[0]
            self.linear1.weight.data = theta[1:].t()
        else:
            self.linear1.weight.data = theta.t()

    def calc_val_loss(self, X, y, reduction='mean'):
        self.eval()
        with torch.no_grad():
            pred = self.forward(X)
            loss = F.mse_loss(pred, y, reduction=reduction)
        if reduction == 'none':
            return loss.detach().numpy().flatten()
        else:
            return loss.item()


class MLP(torch.nn.Module):
    # NOTE: This doesn't exactly fit with our current experiment framework, is just here as an example.
    def __init__(self, input_dim, hidden_dim, n_epochs, lr, scheme):
        super(WLSGD, self).__init__()
        # Set training parameters
        self.n_epochs = n_epochs
        self.lr = lr
        self.scheme = scheme

        # Build the model
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        pred = self.linear1(x)
        pred = F.relu(pred)
        pred = self.linear2(pred)
        pred = F.relu(pred)
        pred = self.linear3(pred)
        return pred

    def fit(self, eta, X, y):
        self.train()
        t = X.shape[0]
        weights = self.scheme.compute_scheme(t, eta)
        opt = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        best_loss = torch.inf
        for i in range(self.n_epochs):
            opt.zero_grad()
            pred = self.forward(X)
            loss = F.mse_loss(pred, y, reduction='none')
            weighted_loss = torch.mean(loss * weights.unsqueeze(1))
            weighted_loss.backward()
            # Snapshot if best epoch
            if weighted_loss < best_loss:
                torch.save(self.state_dict(), 'model.pt')
                best_loss = weighted_loss.item()
            opt.step()
        return best_loss

    def calc_val_loss(self, X, y):
        self.eval()
        with torch.no_grad():
            pred = self.forward(X)
            loss = F.mse_loss(pred, y)
        return loss.item()
