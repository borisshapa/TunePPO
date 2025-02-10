import torch
import typing as tp


class PenalizedPPOStats(tp.NamedTuple):
    """
    Contains Penalized PPO loss statistics (metrics)

    Attributes:
        loss (torch.Tensor): Pure PPO loss.
        policy_loss (torch.Tensor): The policy function loss.
        value_loss (torch.Tensor): The value function loss.
        ratios (torch.Tensor): The ratio between the current and old policy
            probabilities.
        clipfrac (torch.Tensor): The fraction of ratios that were clipped.
        approx_policy_kls (torch.Tensor): Average estimated KL divergence
            between the policy before and after the optimisation step.
        kl_penalty (torch.Tensor): KL-Penalty
    """
    loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    ratios: torch.Tensor
    clipfrac: torch.Tensor
    approx_policy_kls: torch.Tensor
    kl_penalty: torch.Tensor
