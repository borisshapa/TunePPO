import torch

from ppotune.schedulers.scheduler import Scheduler


def linear_scheduler(
    initial_value: float,
    final_value: float,
    num_steps: int,
    param: torch.Tensor,
    **kwargs,
) -> Scheduler:
    linear_shedule_fn = (
        lambda step:
            initial_value +
            (final_value - initial_value) / (num_steps - 1) * step
    )

    return Scheduler(linear_shedule_fn, param)
