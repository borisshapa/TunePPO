import torch
import typing as tp
from torchtune.data import left_pad_sequence, Message


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


class PrefixSuffix(tp.NamedTuple):
    """
    Prefix-Postfix pair to wrap-up messages
    """
    prefix: str
    suffix: str


class PromptTemplate(tp.TypedDict, total=False):
    """
    Chat Template alternative for non-instruct models and tasks. Contains
    prefix-suffix pair to wrap-up message for each role.
    """
    system: PrefixSuffix
    user: PrefixSuffix
    assistant: PrefixSuffix
    ipyhon: PrefixSuffix


def apply_prompt_template(
    template: PromptTemplate,
    messages: tp.List[Message],
    add_generation_prompt: bool = False
) -> str:
    """
    Applies prompt template to a list of messages resulting in single string.
    """
    result = ""
    for message in messages:
        result += (""
            + template[message.role].prefix
            + message.text_content
            + template[message.role].suffix
        )
    if add_generation_prompt:
        result += template["assistant"].prefix

    return result
