from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from torch.utils.data import Dataset

from datasets import load_dataset
from transformers import PreTrainedTokenizerBase


class AlpacaDataset(Dataset):
    """
    Dataset for training PPO with prompts from Alpaca dataset.

    Args:
        tokenizer (PreTrainedTokenizerBase): Tokenizer used by the model
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``. See Hugging
            Face's `API ref <https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset>`_
            for more details.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        source: str,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)

        self._data = self._data.filter(lambda sample: sample["input"] == "")

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        prompt = sample["instruction"]

        tokens = self._tokenizer.apply_chat_template(
            conversation = [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            truncation=True,
            tokenize=True,
        )

        return {"tokens": tokens}


def alpaca_dataset(
    tokenizer: PreTrainedTokenizerBase,
    source: str,
    **load_dataset_kwargs: Dict[str, Any],
) -> AlpacaDataset:
    """
    Builder for Alpaca dataset.

    Args:
        tokenizer (ModelTokenizer): Tokenizer used by the model that implements the ``tokenize_messages`` method.
        source (str): path to dataset repository on Hugging Face. For local datasets,
            define source as the data file type (e.g. "json", "csv", "text") and pass
            in the filepath in ``data_files``. See Hugging Face's ``load_dataset``
            (https://huggingface.co/docs/datasets/en/package_reference/loading_methods#datasets.load_dataset.path)
            for more details.
        filter_fn (Optional[Callable]): callable used to filter the dataset prior to any pre-processing. See
            the Hugging Face `docs <https://huggingface.co/docs/datasets/v2.20.0/process#select-and-filter>`_ for more
            details.
        **load_dataset_kwargs (Dict[str, Any]): additional keyword arguments to pass to ``load_dataset``.

    Returns:
        AlpacaDataset: the configured :class:`~ppotune.datasets.TextCompletionDataset`.
    """

    ds = AlpacaDataset(
        tokenizer=tokenizer,
        source=source,
        **load_dataset_kwargs,
    )

    return ds
