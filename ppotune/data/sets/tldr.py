import typing as tp
import torch.distributed as dist

from ppotune.data.sets.text_completion import (
    TCTransform,
    TextCompletion,
    TextCompletionDataset
)
from torchtune.modules.transforms.tokenizers import ModelTokenizer


class TLDRTransform(TCTransform):
    def __call__(self, sample: tp.Mapping[str, tp.Any]) -> TextCompletion:
        return TextCompletion(
            prompt=sample["prompt"],
            completion=sample["completion"]
        )


def tldr_dataset(
    tokenizer: ModelTokenizer,
    *,
    source: str = "trl-lib/tldr",
    split: str = "train",
    configurations: tp.Optional[list[str]] = None,
    **load_dataset_kwargs: tp.Dict[str, tp.Any],
) -> TextCompletionDataset:

    configuration = configurations[dist.get_rank()] if configurations else None

    return TextCompletionDataset(
        tokenizer=tokenizer,
        source=source,
        sample_transform=TLDRTransform(),
        split=split,
        name=configuration,
        **load_dataset_kwargs
    )
