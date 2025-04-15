from typing import Iterator, Tuple
from torch.nn import Parameter

from torchtune.training import (
    Checkpointer,
    MODEL_KEY,
    set_activation_checkpointing
)
from torchtune.modules import (
    TransformerDecoder,
    TransformerSelfAttentionLayer
)
from torchtune.modules.peft import (
    get_adapter_params,
    set_trainable_params,
)

class LoRAModel:
    def __init__(
        self,
        ckpt: Checkpointer,
        model: TransformerDecoder,
    ) -> None:

        self.ckpt = ckpt
        self.model = model

    def setup(self) -> None:

        state_dict = self.ckpt.load_checkpoint()[MODEL_KEY]

        set_trainable_params(self.model, get_adapter_params(self.model))
        set_activation_checkpointing(
            self.model, auto_wrap_policy={TransformerSelfAttentionLayer}
        )
        # warning: strict=False allows not all model params to be initialized
        # wich is dangerous but is necessary to load with no LoRAs predefined.
        self.model.load_state_dict(state_dict, strict=False)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True, remove_duplicate: bool = True
    ) -> Iterator[Tuple[str, Parameter]]:
        for name, param in self.model.named_parameters(prefix, recurse, remove_duplicate):
            yield name, param
