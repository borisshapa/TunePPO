from abc import ABC, abstractmethod
from omegaconf import DictConfig
from typing import Iterator, Tuple
from torchtune import rlhf

from ppotune.datatypes import AdvantageTrajectoryStats
from ppotune.reward import IRewardModel, LLMRewardModel
from ppotune.utils import append_mask
from ppotune.log import WandbLogger

from torch.nn import Parameter
import torch


logger = WandbLogger()

class IAdvantageModel(ABC):
    """
    Advantage Model Abstract Interface.
    """
    def __init__(self, reward: IRewardModel) -> None:
        self.rm = reward

    def setup(self, cfg: DictConfig) -> None:
        self.rm.setup(cfg.reward)

    @abstractmethod
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs,   # extra arguments for specific reward model
    ) -> AdvantageTrajectoryStats:
        ...

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self.rm.named_parameters(prefix, recurse, remove_duplicate):
            yield name, param

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        for _, param in self.named_parameters(recurse=recurse):
            yield param

    @abstractmethod
    def loss(
        self,
        **kwargs
    ) -> torch.Tensor:
        ...

class GAE(IAdvantageModel):
    """
    Generalized Advantage Estimation.
    Expects llm-based reward model to initialize value model with.
    """
    def __init__(
        self,
        reward: LLMRewardModel,
        gamma: float,
        lmbda: float,
        value_coeff: float,
        value_clip_range: float
    ) -> None:

        super().__init__(reward)

        self.gamma = gamma
        self.lmbda = lmbda
        self.value_coeff = value_coeff
        self.value_clip_range = value_clip_range

    @torch.no_grad()
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs,   # extra arguments for specific reward model
    ) -> AdvantageTrajectoryStats:

        rewards = self.rm(
            tokens,
            causal_mask,
            position_ids,
            responses_pad_mask,
            **kwargs,
        )
        value_pad_mask = append_mask(
            responses_pad_mask,
        )
        values = self.estimate_values(
            tokens,
            causal_mask,
            position_ids,
            value_pad_mask
        )
        advantages, returns = rlhf.estimate_advantages(
            values,
            rewards,
            self.gamma,
            self.lmbda,
            masks=~responses_pad_mask
        )
        return AdvantageTrajectoryStats(
            advantages=advantages,
            values=values,
            returns=returns,
            rewards=rewards
        )

    def estimate_values(
        self,
        tokens:         torch.Tensor, # B x (Q + R)
        causal_mask:    torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:   torch.Tensor, # B x (Q + R)
        padding_mask:   torch.Tensor, # B x R
    ) -> torch.Tensor: # B x R
        """
        Value function inference
        """
        values = self.rm.scorer.model(
            tokens,
            input_pos=position_ids,
            mask=causal_mask
        ).squeeze(-1) # single output at "vocab" dim
        queries_len = tokens.shape[1] - padding_mask.shape[1]
        values = values[:, queries_len - 1 : -1]
        values[padding_mask] = 0.0
        return values

    def loss(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        inference_values:   torch.Tensor, # B x R
        inference_returns:  torch.Tensor, # B x R
    ) -> torch.Tensor:
        """
        Execute value function optimisation step
        """
        value_pad_mask = append_mask(responses_pad_mask)
        values = self.estimate_values(
            tokens,
            causal_mask,
            position_ids,
            value_pad_mask
        )
        values_clipped = torch.clamp(
            values,
            inference_values - self.value_clip_range,
            inference_values + self.value_clip_range,
        )
        loss = torch.maximum(
            (values - inference_returns) ** 2, (values_clipped - inference_returns) ** 2
        )
        loss = 0.5 * rlhf.masked_mean(loss, ~value_pad_mask)

        logger.collect("value_loss", loss)
        return self.value_coeff * loss


class GRAE(IAdvantageModel):
    """
    Group Relative Advantage Estimation.
    """
    def __init__(
        self,
        reward: IRewardModel,
        group_size: int
    ) -> None:

        super().__init__(reward)
        self.group_size = group_size

    @torch.no_grad()
    def __call__(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        responses_pad_mask: torch.Tensor, # B x R
        **kwargs,   # extra arguments for specific reward model
    ) -> AdvantageTrajectoryStats:

        rewards = self.rm(
            tokens,
            causal_mask,
            position_ids,
            responses_pad_mask,
            **kwargs
        )
        group_scores = rewards.reshape(-1, self.group_size)
        group_advantages = (group_scores - group_scores.mean(1, keepdim=True)) / (
            group_scores.std(1, keepdim=True) + 1e-4
        )
        advantages = group_advantages.flatten().unsqueeze(-1)

        return AdvantageTrajectoryStats(
            advantages=advantages,
            values=torch.zeros_like(advantages), # irrelevant
            returns=torch.zeros_like(advantages), # irrelevant
            rewards=rewards
        )

    def loss(self, **kwargs) -> torch.Tensor:
        loss = torch.tensor(0.0)
        logger.collect("value_loss", loss)
        return loss
