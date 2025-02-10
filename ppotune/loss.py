import torch
import torch.nn as nn

from torchtune.rlhf.rewards import masked_mean

class KLPenalty(nn.Module):
    """
    KL-Penalty module. Provides KL approximation of log probabilities batch
    according to [Schulman et al. 2020](http://joschu.net/blog/kl-approx.html)
    """
    def __init__(self, coeff: float) -> None:
        super().__init__()
        self._coeff = coeff

    def forward(
        self,
        lhs_logprobs: torch.Tensor,
        rhs_logprobs: torch.Tensor,
        padding_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes coeff * KL(lhs | rhs)
        """
        kl = torch.exp(rhs_logprobs - lhs_logprobs) - (rhs_logprobs - lhs_logprobs) - 1
        return self._coeff * masked_mean(kl, padding_masks)
