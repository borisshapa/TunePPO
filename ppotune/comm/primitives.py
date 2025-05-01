import typing as tp
import torch
import torch.distributed as dist


# ------------------ Distributed Communication Primitives ------------------- #
#
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


# ------------------------ Distributed Map Primitives ----------------------- #
#
def uniform(
    self_preference: tp.Optional[float] = None
) -> torch.Tensor:
    """
    Prodices uniform weights over peer ranks with optional prevalence to self.
    """
    if dist.get_world_size() == 1:
        return torch.tensor([1.0])

    if self_preference is None:
        self_preference = 1.0 / dist.get_world_size()

    other_preference = (1 - self_preference) / (dist.get_world_size() - 1)
    weights = other_preference * torch.ones(dist.get_world_size())
    weights[dist.get_rank()] = self_preference
    return weights

def softmax(
    weight: tp.Optional[torch.Tensor] = None,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Computes softmax over peer weights.
    """
    if weight is None:
        return uniform()

    peer_weights = torch.tensor(all_gather_even(weight)) / temperature
    return torch.softmax(peer_weights, dim=0).tolist()
