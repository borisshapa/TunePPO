import typing as tp

import torch
import torch.distributed as dist

from torchtune.modules import TransformerDecoder

from ppotune.log import WandbLogger


log = WandbLogger()

# ------------------ Distributed Communication Primitives ------------------- #
def all_gather_even(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Performs torch.distributed.all_gather and returns the output list. Tensors
    have to be of the same shape.
    """
    peer_tensors = [
        torch.empty_like(tensor) for _ in range(dist.get_world_size())
    ]
    dist.all_gather(peer_tensors, tensor)
    return peer_tensors

def all_gather_shapes(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Collects shapes of tensors in distributed system. Tensors have to be of
    the same number of dimensions.
    """
    shape = torch.tensor(tensor.shape, dtype=torch.int64, device=tensor.device)
    return all_gather_even(shape)

def all_gather_uneven(tensor: torch.Tensor) -> list[torch.Tensor]:
    """
    Collects unevenly shaped tensors in distributed system. Tensors have to be
    of the same number of dimensions.
    """
    peer_shapes = all_gather_shapes(tensor)
    peer_tensors = [
        torch.empty(shape.tolist(), dtype=tensor.dtype, device=tensor.device)
        for shape in peer_shapes
    ]
    dist.all_gather(peer_tensors, tensor)
    return peer_tensors

def all_to_all(tensors: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Peforms torch.distributed.all_to_all and returns the output list. Input
    tensor lists has to be of the same shape configuration across all
    processes i.e tensors[rank].shape == const for any rank.
    """
    reference = tensors[dist.get_rank()]
    output = [torch.empty_like(reference) for _ in tensors]
    dist.all_to_all(output, tensors)
    return output


# --------------------- Distributed Special Utilities ----------------------- #
class WeightedMean:
    def __init__(self, weights: tp.List[float]) -> None:
        self._weights = weights

    @property
    def weights(self) -> torch.Tensor:
        weights = torch.tensor(self._weights)
        return weights / weights.sum()

    def __call__(
        self,
        tensors: tp.Iterable[torch.Tensor],
    ) -> torch.Tensor:
        tensors = torch.stack(tensors)
        weights = self.weights.to(tensors[0].device)
        log.collect("self_preference", weights[dist.get_rank()])
        for _ in range(tensors.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (weights * tensors).sum(dim=0)


def mean() -> WeightedMean:
    weights = [1. / dist.get_world_size()] * dist.get_world_size()
    return WeightedMean(weights)


def weighted_mean(
    weights: tp.Optional[tp.List[float]] = None
) -> WeightedMean:
    if weights is None:
        return mean()

    return WeightedMean(weights)


def softmax(
    weights: tp.Optional[tp.List[float]] = None,
    temperature: tp.Optional[float] = None
) -> WeightedMean:
    if weights is None:
        return mean()

    if temperature is not None:
        weights /= temperature

    weights = torch.softmax(torch.tensor(weights), dim=0)

    return WeightedMean(weights.tolist())


def distributed_weighted_mean(
    self_weight: tp.Optional[torch.Tensor]
) -> WeightedMean:
    if self_weight is None:
        return mean()

    peer_weights = all_gather_even(self_weight)
    return weighted_mean(peer_weights)


def distributed_softmax(
    self_weight: tp.Optional[torch.Tensor],
    temperature: tp.Optional[float] = None,
) -> WeightedMean:
    if self_weight is None:
        return mean()

    peer_weights = all_gather_even(self_weight)
    return softmax(peer_weights, temperature)


def self_preferred_mean(
    self_preference: tp.Optional[float] = None
) -> WeightedMean:

    if self_preference is None:
        return mean()

    if dist.get_world_size() == 1:
        return mean()

    other_preference = (1 - self_preference) / (dist.get_world_size() - 1)
    weights = [other_preference] * dist.get_world_size()
    weights[dist.get_rank()] = self_preference
    return weighted_mean(weights)


def self_preferred_weighted_mean(
    self_preference: tp.Optional[float] = None,
    weights: tp.Optional[tp.List[float]] = None,
) -> WeightedMean:
    self_preferred_weights = self_preferred_mean(self_preference).weights
    return weighted_mean((self_preferred_weights * torch.tensor(weights)).tolist())


def self_preferred_distributed_weighted_mean(
    self_preference: tp.Optional[float] = None,
    self_weight: tp.Optional[torch.Tensor] = None,
) -> WeightedMean:
    self_preferred_weights = self_preferred_mean(self_preference).weights
    distributed_weights = distributed_weighted_mean(self_weight).weights
    return weighted_mean((self_preferred_weights * distributed_weights).tolist())

# -----------------------------------------------------------------------------

def self_preferred_distributed_softmax(
    self_preference: tp.Optional[float] = None,
    self_weight: tp.Optional[torch.Tensor] = None,
    temperature: tp.Optional[float] = None,
) -> WeightedMean:
    self_preferred_weights = self_preferred_mean(self_preference).weights
    distributed_softmax_weights = distributed_softmax(self_weight, temperature).weights
    return weighted_mean(
        (self_preferred_weights * distributed_softmax_weights).tolist()
    )

# ----------------------- Distributed Policy Mixture ------------------------ #
class DistributedPolicyMixture:
    """
    Implements interprocess communication and delivers a mixtxture of policies.
    """
    def __init__(
        self,
        local_policy: TransformerDecoder,
        reducer: tp.Optional[WeightedMean] = None,
    ) -> None:
        self._policy = local_policy
        if reducer is None:
            reducer = weighted_mean()
        self._reducer=reducer

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Follows the interface of TransformerDecoder forward method. Use it as a
        more comprehensive reference.
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (torch.Tensor): used to mask the scores after the query-key
                multiplication and before the softmax. This parameter is
                required during inference.
            input_pos (torch.Tensor): contains the position ids of each token.
                During training, this is used to indicate the positions of each
                token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current
                token. This parameter is required during inference.

        Returns:
            torch.Tensor: output tensor with shape ``[b x s x v]``

        Shape notation:
            - b: batch size
            - s: token sequence length
            - v: vocab size
        """
        peer_tokens = all_gather_uneven(tokens)
        peer_masks = all_gather_uneven(mask)
        peer_pos = all_gather_uneven(input_pos)

        peer_logits_requested = [
            self._policy(tokens, input_pos=pos, mask=mask)
            for tokens, pos, mask in zip(peer_tokens, peer_pos, peer_masks)
        ]
        peer_logits_responded = all_to_all(peer_logits_requested)
        return self._reducer(peer_logits_responded)

    def __call__(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes forward pass.
        """
        return self.forward(tokens, mask, input_pos)
