import typing as tp

from types import MethodType

from torchtune.modules.transforms.tokenizers import ModelTokenizer
from torchtune.data import Message
from transformers import AutoTokenizer
from transformers import PreTrainedTokenizerBase


def messages_to_conversation(
    messages: tp.List[Message]
) -> tp.List[tp.Dict[str, str]]:

    conversation = [
        {"role": message.role, "content": message.content}
        for message in messages
    ]

    return conversation


def tokenize_messages(
    self: PreTrainedTokenizerBase,
    messages: tp.List[Message],
    add_generation_prompt: bool = False,
    **kwargs: tp.Dict[str, tp.Any]
) -> tp.List[int]:

    return self.apply_chat_template(
        conversation = messages_to_conversation(messages),
        add_generation_prompt=add_generation_prompt,
        truncate=True,
        tokenize=True,
        **kwargs
    )


def hf_tokenizer(
    path: str,
    pad_token: str | None = None,
    max_seq_len: int | None = None,
) -> ModelTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(
        path,
        pad_token = pad_token,
        model_max_length = max_seq_len
    )

    tokenizer.pad_id = tokenizer.pad_token_id
    tokenizer.eos_id = tokenizer.eos_token_id
    tokenizer.tokenize_messages = MethodType(tokenize_messages, tokenizer)

    return tokenizer
