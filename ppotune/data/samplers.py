import typing as tp
import torch

from torch.utils.data import Sampler
from itertools import chain, repeat


class PermutationSampler(Sampler[int]):
    """
    Permutes elements in range(num_samples) randomly.

    Args:
        num_samples (sequence): num samples to permute.
        generator (Generator): Generator used in sampling.
    """
    def __init__(
        self,
        num_samples: int,
        generator=None
    ) -> None:
        self.num_samples = num_samples
        self.generator = generator

    def __iter__(self) -> tp.Iterator[int]:
        for idx in torch.randperm(self.num_samples, generator=self.generator):
            yield idx

    def __len__(self) -> int:
        return self.num_samples


class RepeatedSampler(Sampler[int]):
    """
    Wrap any `Sampler[int]` and iterate through it `num_epochs` times,
    producing one flat stream of indices.

    Args
    ----
    base_sampler : Sampler[int]
        The sampler whose order you want to repeat (e.g. RandomSampler,
        SequentialSampler, WeightedRandomSampler, ...).
    num_epochs : int
        How many complete passes over `base_sampler` to emit.
    """
    def __init__(
        self,
        base_sampler: Sampler[int],
        num_epochs: int
    ) -> None:
        self.base_sampler = base_sampler
        self.num_epochs   = num_epochs

    def __iter__(self) -> tp.Iterator[int]:
        for _ in range(self.num_epochs):
            yield from iter(self.base_sampler)

    def __len__(self) -> int:
        return self.num_epochs * len(self.base_sampler)


class GroupedSampler(Sampler[int]):
    """
    Emit each index from `base_sampler` consecutively `group_factor` times.

    Example
    -------
    base_sampler   -> [0, 3, 2]
    group_factor=2 ->  [0, 0, 3, 3, 2, 2]

    Parameters
    ----------
    base_sampler : Sampler[int]
        Any order-producing sampler (RandomSampler, SequentialSampler, ...).
    group_factor : int
        Number of contiguous duplicates per index (>= 1).
    """
    def __init__(
        self,
        base_sampler: Sampler[int],
        group_factor: int = 1
    ) -> None:
        self.base_sampler = base_sampler
        self.group_factor = group_factor

    def __iter__(self) -> tp.Iterator[int]:
        return chain.from_iterable(
            repeat(idx, self.group_factor) for idx in self.base_sampler
        )

    def __len__(self) -> int:
        return self.group_factor * len(self.base_sampler)
