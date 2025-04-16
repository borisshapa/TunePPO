from omegaconf import DictConfig

import torch
import torch.nn as nn
import typing as tp

from torchtune.training import (
    Checkpointer,
    MODEL_KEY,
    set_activation_checkpointing
)
from torchtune.modules import (
    TransformerDecoder,
    TransformerSelfAttentionLayer,
    local_kv_cache
)
from torchtune.modules.peft import (
    get_adapter_params,
    set_trainable_params,
)
from torchtune import generation
from torchtune import utils
from ppotune.peft import (
    get_merged_adapter_state_dict,
    get_adapter_config
)


class LoRAModel(nn.Module):
    def __init__(
        self,
        ckpt: Checkpointer,
        model: TransformerDecoder,
    ) -> None:

        super().__init__()
        self.ckpt = ckpt
        self.model = model
        self._dtype=torch.get_default_dtype()
        self._device=utils.get_device("cuda")

    def setup(self, cfg: DictConfig) -> None:

        self._adapter_config = get_adapter_config(cfg.model)
        set_trainable_params(self.model, get_adapter_params(self.model))

        state_dict = self.ckpt.load_checkpoint()[MODEL_KEY]
        set_activation_checkpointing(
            self.model, auto_wrap_policy={TransformerSelfAttentionLayer}
        )
        # warning: strict=False allows not all model params to be initialized
        # wich is dangerous but is necessary to load with no LoRAs predefined.
        self.model.load_state_dict(state_dict, strict=False)

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:

        logits = self.model(
            tokens,
            mask=mask,
            input_pos=input_pos
        )
        return logits

    def save_checkpoint(self, epoch: int = 0) -> None:
        self.ckpt.save_checkpoint(
            state_dict=get_merged_adapter_state_dict(
                module=self.model,
                module_adapter_config=self._adapter_config
            ),
            epoch=epoch
        )

class GenerativeLoRAModel(LoRAModel):
    def __init__(
        self,
        ckpt: Checkpointer,
        model: TransformerDecoder,
        max_seq_len: int,
        max_generated_tokens: int,
        generation_batch_size: int,
        temperature: float,
        top_k: tp.Optional[int] = None,
        pad_id: int = 0,
        rng: tp.Optional[torch.Generator] = None,
    ) -> None:

        super().__init__(ckpt, model)
        self._max_seq_len = max_seq_len
        self._max_generated_tokens = max_generated_tokens
        self._generation_batch_size = generation_batch_size
        self._temperature = temperature
        self._top_k = top_k
        self._pad_id = pad_id
        self._rng = rng

    def setup(self, cfg: DictConfig) -> None:

        super().setup(cfg)
        self.cache_ctx_manager = lambda: local_kv_cache(
            self.model,
            batch_size=self._generation_batch_size,
            dtype=self._dtype,
            decoder_max_seq_len=self._max_seq_len + self._max_generated_tokens,
            device=self._device
        )

    def generate(
        self, prompt: torch.Tensor
    ) -> tp.Tuple[torch.Tensor, torch.Tensor]:

        with self.cache_ctx_manager():
            tokens, logits = generation.generate(
                model=self.model,
                prompt=prompt,
                max_generated_tokens=self._max_generated_tokens,
                temperature=self._temperature,
                top_k=self._top_k,
                pad_id=self._pad_id,
                rng=self._rng,
            )
        return tokens, logits
