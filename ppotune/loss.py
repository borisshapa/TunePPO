import torch
import torch.nn as nn

from torchtune.rlhf.rewards import masked_mean

from ppotune.volatile import VolatileFloat
from ppotune.log import WandbLogger


logger = WandbLogger()

class KLPenalty(nn.Module):
    """
    KL-Penalty module. Provides KL approximation of log probabilities batch
    according to [Schulman et al. 2020](http://joschu.net/blog/kl-approx.html)
    """
    def __init__(
        self,
        coeff: float | VolatileFloat
    ) -> None:
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
        coeff = torch.tensor(float(self._coeff))

        per_token_kl = torch.exp(rhs_logprobs - lhs_logprobs) - (rhs_logprobs - lhs_logprobs) - 1
        kl_penalty = coeff * masked_mean(per_token_kl, padding_masks)

        logger.collect("kl_penalty", kl_penalty)
        logger.collect("kl_coeff", coeff)

        return kl_penalty
