import typing as tp

from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler


class StickyDistributedSampler(Sampler[int]):
    """
    DistributedSampler wrapper that sticks at each element for predefined
    number of repetitions
    """
    def __init__(
        self,
        dataset: Dataset,
        num_duplicates: int,
        num_replicas: tp.Optional[int] = None,
        rank: tp.Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ) -> None:

        self.sampler = DistributedSampler(
            dataset=dataset,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed,
            drop_last=drop_last
        )
        self.num_duplicates = num_duplicates

    def __iter__(self) -> tp.Iterator[int]:
        indices = list(iter(self.sampler))
        duped_indices = [
            idx for idx in indices for _ in range(self.num_duplicates)
        ]
        return iter(duped_indices)

    def __len__(self) -> int:
        return self.num_duplicates * len(self.sampler)


    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for this sampler.
        """
        self.sampler.set_epoch(epoch)
