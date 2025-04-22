import torch

from ppotune.schedulers.scheduler import Scheduler


def constant_scheduler(value: float, param: torch.Tensor) -> Scheduler:
    constant_scheduler_fn = lambda _: value

    return Scheduler(constant_scheduler_fn, param)
