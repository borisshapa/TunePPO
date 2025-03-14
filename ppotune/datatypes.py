import torch
import typing as tp


class ExtendedTrajectory(tp.NamedTuple):
    """
    Contains a collection of tensors describing a generated trajectory during RLHF

    Attributes:
        query_responses (torch.Tensor): (query, response) pairs
            shape [b, context_length + max_generated_tokens]
        logprobs (torch.Tensor): log probabilities of the generated responses with
            shape [b, max_generated_tokens]
        ref_logprobs (torch.Tensor): log probs of the generated responses w.r.t. reference policy
            shape [b, max_generated_tokens]
        values (torch.Tensor): value estimates of the generated responses with
            shape [b, max_generated_tokens]
        masks (torch.Tensor): attention masks for input ids-generated responses pairs
            shape [b, context_length + max_generated_tokens, context_length + max_generated_tokens]
        position_ids (torch.Tensor): position IDs for input ids-generated responses pairs
            shape [b, context_length + max_generated_tokens]
        response_padding_masks (torch.Tensor): padding masks for the truncated and padded generated responses
            shape [b, max_generated_tokens]
        value_padding_masks (torch.Tensor): padding masks for the values with
            shape [b, max_generated_tokens]
        value_seq_idxs (torch.Tensor): indexes of the token
            after the last valid (non-padding) token in the responses with
            shape [b]
        scores (torch.Tensor): scores from the reward model with
            shape [b]
        seq_lens (torch.Tensor): sequence lengths of truncated generated responses with
            shape [b]
        kl (torch.Tensor) kl divergence between policy and reference policy logprobs with
            shape [b, response_len]
        kl_reward (torch.Tensor) kl divergence scaled by kl coefficient with
            shape [b, response_len]
        advantages (torch.Tensor): the estimated advantages with
            shape: [b, response_len]
        returns (torch.Tensor): the estimated returns with
            shape: [b, response_len]
    """
    query_responses: torch.Tensor
    logprobs: torch.Tensor
    ref_logprobs: torch.Tensor
    values: torch.Tensor
    masks: torch.Tensor
    position_ids: torch.Tensor
    response_padding_masks: torch.Tensor
    value_padding_masks: torch.Tensor
    value_seq_idxs: torch.Tensor
    scores: torch.Tensor
    seq_lens: torch.Tensor
    kl: torch.Tensor
    kl_rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor


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


class PPOAdvantageModelResult(tp.NamedTuple):
    values: torch.Tensor
    value_pad_mask: torch.Tensor
    value_last_pos: torch.Tensor
    scores: torch.Tensor
    score_pos: torch.Tensor
    kl: torch.Tensor
    kl_rewards: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
