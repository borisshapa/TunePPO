import typing as tp

import torch

class Scheduler:
    """
    Base class for schedulers.
    Args:
        schedule_fn:
            function to get parameter values based on current step.
            Accepts:
                step number
            Returns:
                new parameter value.
        param:
            scheduled parameter tensor.
    """
    def __init__(
        self,
        schedule_fn: tp.Callable,
        param: torch.Tensor,
    ) -> None:
        self._param = param
        self._schedule_fn = schedule_fn
        self._cur_step = 0

    def step(self) -> None:
        new_value = self._schedule_fn(self._cur_step)
        self._param.fill_(new_value)
        self._cur_step += 1
