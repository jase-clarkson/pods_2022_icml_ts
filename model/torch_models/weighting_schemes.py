import torch
import torch.nn.functional as F


class ExpDecay:
    eta_dim = 1

    @staticmethod
    def compute_scheme(t, eta):
        eta_pos = torch.abs(eta.expand(t))
        weights = torch.exp(-eta_pos * (torch.arange(t-1, -1, -1)))
        delta = 1 if torch.sum(weights) < 1e-12 else 0
        return (weights / (torch.sum(weights) + delta)).float()


class MixedDecay:
    eta_dim = 3

    @staticmethod
    def compute_scheme(t, eta):
        decay_seq = torch.arange(t-1, -1, -1)  # t-dim (list)
        exp_seq = -1 * decay_seq
        sqexp_seq = -1 * decay_seq ** 2
        power_seq = -1 * torch.log(decay_seq + 1)
        weights = torch.stack([exp_seq, sqexp_seq, power_seq])
        weights = weights * torch.abs(eta[:, None])
        weights = torch.exp(weights.sum(dim=0)).float()
        delta = 1 if torch.sum(weights) < 1e-12 else 0
        return (weights / (torch.sum(weights) + delta)).float()
