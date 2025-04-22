import typing as tp

import torch

class Scheduler:
    """
    Base class for schedulers.
    Args:
        param:
            scheduled parameter tensor.
        schedule_fn:
            function to get parameter values based on current step.
            Accepts:
                step number
            Returns:
                new parameter value.
    """
    def __init__(
        self,
        schedule_fn: tp.Callable
    ) -> None:
        self._param = None
        self._schedule_fn = schedule_fn
        self._cur_step = 0

    def add_param(self, param: torch.Tensor) -> None:
        self._param = param

    def step(self) -> None:
        assert self._param is not None

        new_value = self._schedule_fn(self._cur_step)
        self._param.fill_(new_value)
        self._cur_step += 1

