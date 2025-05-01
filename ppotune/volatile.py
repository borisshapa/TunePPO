import typing as tp


class VolatileFloat(tp.Iterator[float]):
    """
    Volatile float that changes it's value according to some predefined
    schedule whenever is accessed.
    """
    def __init__(
        self,
        schedule: tp.Iterator[float],
    ) -> None:

        self._schedule = schedule

    def __iter__(self) -> tp.Self:
        return self

    def __next__(self) -> float:
        return next(self._schedule)

    def __float__(self) -> float:
        return next(self)


def constant_schedule(
    value: float
) -> tp.Iterator[float]:
    """
    Yields the same value all the time.
    """
    while True:
        yield value


def linear_schedule(
    start: float,
    end: float,
    steps: int
) -> tp.Iterator[float]:
    """
    Yields values from start to end linearly over the number of steps given.
    """
    delta = (end - start) / (steps - 1)
    for step in range(steps):
        yield start + step * delta


def const(
    value: float
) -> VolatileFloat:
    """
    Builds constant volatile float.
    """
    return VolatileFloat(constant_schedule(value))


def linearly_scheduled(
    start: float,
    end: float,
    steps: int
) -> VolatileFloat:
    """
    Builds linearly scheduled volatile float.
    """
    return VolatileFloat(linear_schedule(start, end, steps))
