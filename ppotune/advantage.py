import torch
import torch.nn as nn

from torchtune import rlhf
from torchtune.training import get_unmasked_sequence_lengths
from torchtune.modules.peft import disable_adapter

from ppotune.datatypes import PPOAdvantageModelResult


class PPOAdvatageModel(nn.Module):
    """
    Original OpenAI style generalized advantage estimation with kl penalized
    LLM reward model.
    """
    def __init__(
        self,
        scorer: nn.Module,
        gamma: float,
        lmbda: float,
        kl_coeff: float,
        penalise_no_eos: bool = True,
        reward_penalty: int = -3,
        min_response_len: int = 0,
        max_response_len: int = 1024,
    ) -> None:

        super().__init__()

        self.scorer = scorer
        self.gamma = gamma
        self.lmbda = lmbda
        self.kl_coeff = kl_coeff
        self.penalise_no_eos = penalise_no_eos
        self.reward_penalty = reward_penalty
        self.min_response_len = min_response_len
        self.max_response_len = max_response_len

    from torchtune.rlhf.loss import PPOLoss
    def forward(
        self,
        tokens:             torch.Tensor, # B x (Q + R)
        causal_mask:        torch.Tensor, # B x (Q + R) x (Q + R)
        position_ids:       torch.Tensor, # B x (Q + R)
        queries_pad_mask:   torch.Tensor, # B x Q
        responses_pad_mask: torch.Tensor, # B x R
        gen_logprobs:       torch.Tensor, # B x R
        ref_logprobs:       torch.Tensor, # B x R
    ) -> PPOAdvantageModelResult:

        queries_len = queries_pad_mask.shape[1]

        # estimate values
        values = self.scorer(
            tokens,
            input_pos=position_ids,
            mask=causal_mask
        ).squeeze(-1) # single output at "vocab" dim
        values = values[:, queries_len - 1 : -1]

        # estimate scores with reward model
        with disable_adapter(self.scorer):
            scores = self.scorer(
                tokens,
                input_pos=position_ids,
                mask=causal_mask
            )

        # the scores from the reward model are the logits for the last non-padding token
        response_last_pos = get_unmasked_sequence_lengths(responses_pad_mask)
        scores = scores.gather(1, (response_last_pos + queries_len)[:, None, None]).squeeze(
            (-1, -2)
        )
        # if configured, apply penalties for no EOS or too short responses
        if self.penalise_no_eos or self.min_response_len:
            reward_penalty_mask = rlhf.get_reward_penalty_mask(
                responses_pad_mask,
                response_last_pos,
                self.penalise_no_eos,
                self.min_response_len,
            )
            scores[reward_penalty_mask] = self.reward_penalty

        # values are masked out *after* the last valid token in the response
        value_last_pos = torch.where(
            (response_last_pos > 0) & (response_last_pos < self.max_response_len - 1),
            response_last_pos + 1,
            response_last_pos,
        )
        value_pad_mask = responses_pad_mask.clone()
        value_pad_mask = value_pad_mask.scatter_(
            1, value_last_pos.unsqueeze(-1), False
        ) # unmask last value entry
        values[value_pad_mask] = 0.0

        # get the trajectory rewards based on:
        # - the divergence between the current policy and the reference policy
        # - the scores from the reward model
        rewards, kl, kl_rewards = rlhf.get_rewards_ppo(
            scores,
            gen_logprobs,
            ref_logprobs,
            self.kl_coeff,
            value_last_pos
        )

        # estimate the advantages with GAE
        advantages, returns = rlhf.estimate_advantages(
            values,
            rewards,
            self.gamma,
            self.lmbda,
            masks=~responses_pad_mask
        )

        return PPOAdvantageModelResult(
            values=values,
            value_pad_mask=value_pad_mask,
            value_last_pos=value_last_pos,
            scores=scores,
            score_pos=response_last_pos,
            kl=kl,
            kl_rewards=kl_rewards,
            advantages=advantages,
            returns=returns
        )
