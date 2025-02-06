import typing as tp
import torch


@torch.no_grad()
def grad_norm(parameters: tp.Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Computes gradient l2-norm of parameters given.
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm
