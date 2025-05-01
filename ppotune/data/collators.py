import torch
import typing as tp
from torchtune.data import left_pad_sequence


class LeftPadCollator:
    def __init__(
        self,
        tokens_key: str = "tokens",
        pad_token: int = 0
    ) -> None:
        self.tokens_key = tokens_key
        self.pad_token = pad_token

    def __call__(
        self,
        batch: tp.List[tp.Dict[str, tp.List[int]]],
    ) -> tp.Dict[str, tp.Union[torch.Tensor, tp.List[str]]]:

        non_token_keys = [k for k in batch[0].keys() if k != self.tokens_key]
        collated_batch = {
            k : [row[k] for row in batch] for k in non_token_keys
        }
        collated_tokens = left_pad_sequence(
            [torch.tensor(row[self.tokens_key]) for row in batch],
            batch_first=True,
            padding_value=self.pad_token
        )
        collated_batch[self.tokens_key] = collated_tokens.long()
        return collated_batch
