import typing as tp
from torchtune.data import Message


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
