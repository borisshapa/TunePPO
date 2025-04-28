import torch
import torch.distributed as dist

from torch.utils.data import DataLoader, Dataset, Subset
from torchtune.modules.transforms.tokenizers import ModelTokenizer

from ppotune.data.collators import LeftPadCollator
from ppotune.data.samplers import (
    RepeatedSampler,
    GroupedSampler,
    PermutationSampler
)


def subset(
    dataset: Dataset,
    num_samples: int,
    generator: torch.Generator = None,
) -> Subset:
    """
    Selects random subset of the dataset.
    """
    assert num_samples <= len(dataset)

    indices = torch.randperm(len(dataset), generator=generator)[:num_samples]
    subset = Subset(dataset, indices)
    return subset


def distributed_subset(
    dataset: Dataset,
    num_samples: int,
    generator: torch.Generator,
) -> Subset:
    """
    Split dataset for each rank evenly.
    """
    assert dist.get_world_size() * num_samples <= len(dataset)

    indices = torch.randperm(len(dataset), generator=generator)
    local_offset = dist.get_rank() * num_samples
    local_indices = indices[local_offset : local_offset + num_samples]
    local_subset = Subset(dataset, local_indices)
    return local_subset


def dataloader(
    tokenizer: ModelTokenizer,
    dataset: Dataset,
    *
    num_steps: int,
    batch_size: int,
    group_size: int = 1,
    num_epochs: int = 1,
    seed: int = 0,
) -> DataLoader:
    """
    Basic dataloader.
    """
    assert batch_size % group_size == 0
    raw_batch_size = batch_size // group_size
    num_samples = num_steps * raw_batch_size

    generator = torch.Generator().manual_seed(seed + dist.get_rank() + 1)
    dataset = subset(dataset, num_samples, generator)

    sampler = RepeatedSampler(
        GroupedSampler(
            PermutationSampler(len(dataset), generator=generator),
            group_factor=group_size,
        ),
        num_epochs=num_epochs
    )
    collator = LeftPadCollator(
        tokens_key="tokens",
        pad_token=tokenizer.pad_id
    )
    return DataLoader(
        dataset=dataset,
        sampler=sampler,
        batch_size=raw_batch_size,
        drop_last=True,
        collate_fn=collator
    )


def distributed_dataloader(
    tokenizer: ModelTokenizer,
    dataset: Dataset,
    *
    num_steps: int,
    batch_size: int,
    group_size: int = 1,
    num_epochs: int = 1,
    seed: int = 0,
) -> DataLoader:
    """
    Dataloader for distributed setup.
    """
    assert batch_size % group_size == 0
    raw_batch_size = batch_size // group_size
    num_samples = num_steps * raw_batch_size

    generator = torch.Generator().manual_seed(seed)
    local_dataset = distributed_subset(dataset, num_samples, generator)

    return dataloader(
        tokenizer,
        local_dataset,
        num_steps=num_steps,
        batch_size=batch_size,
        group_size=group_size,
        num_epochs=num_epochs,
        seed=seed,
    )
