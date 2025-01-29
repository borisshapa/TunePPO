# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import sys
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional, Tuple, Iterable, Iterator
from warnings import warn

import torch
import torch.distributed as dist
from omegaconf import DictConfig, ListConfig
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler
from torchao.dtypes import NF4Tensor
from torchtune import config, generation, modules, rlhf, training, utils
from torchtune.data import padded_collate
from torchtune.datasets import ConcatDataset
from torchtune.modules.peft import (
    disable_adapter,
    get_adapter_params,
    get_lora_module_names,
    get_merged_lora_ckpt,
    set_trainable_params,
)
from torchtune.modules.peft import LoRALinear
from torchtune.modules.peft.lora import to_nf4
from torchtune.modules.tokenizers import ModelTokenizer
from torchtune.recipe_interfaces import FTRecipeInterface
from torchtune.rlhf import PPOStats, Trajectory
from torchtune.training.checkpointing import Checkpointer
from torchtune.training.metric_logging import MetricLoggerInterface
from torchtune.utils import log_rank_zero
from tqdm import tqdm

log = utils.get_logger("DEBUG")

@torch.no_grad()
def grad_norm(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Computes gradient l2-norm of parameters given.
    """
    total_norm = 0.0
    for param in parameters:
        if param.grad is not None:
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

def get_adapter_config(cfg_model: DictConfig) -> Dict[str, Any]:
    """
    Retrieves adapter config from model config suitable for checkpointing.
    """
    return {
        "r": cfg_model.lora_rank,
        "lora_alpha": cfg_model.lora_alpha,
        "target_modules": get_lora_module_names(
            list(cfg_model.lora_attn_modules),
            getattr(cfg_model, "apply_lora_to_mlp", False),
            getattr(cfg_model, "apply_lora_to_output", False)
        ),
        "peft_type": "LORA",
    }

def get_merged_adapter_state_dict(
    module: nn.Module,
    module_adapter_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Constructs model merged with adapter state dict.
    """
    return {
        training.MODEL_KEY: get_merged_lora_ckpt(
            state_dict  = {k: v.cpu() for k, v in module.state_dict().items()},
            rank        = module_adapter_config["r"],
            alpha       = module_adapter_config["lora_alpha"],
        )
    }

def get_lora_modules(model: nn.Module) -> Iterator[LoRALinear]:
    """
    Returns an iterator over all LoRA modules in the network.
    """
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield module

@torch.no_grad()
def merge_lora_adapter(model: nn.Module) -> nn.Module:
    """
    Merges (Q)LoRA adapters into base model in-place.
    """
    for m in get_lora_modules(model):
        if isinstance(m.weight, NF4Tensor):
            dequantw = m.weight.get_original_weight()
            dequantw += (m.alpha / m.rank) * m.lora_b.weight @ m.lora_a.weight
            m.weight = nn.Parameter(to_nf4(dequantw))
        else:
            m.weight += (m.alpha / m.rank) * m.lora_b.weight @ m.lora_a.weight

    return model

@torch.no_grad()
def clear_lora_adapter(model: nn.Module) -> nn.Module:
    """
    Reinitialize LoRA adapters.
    """
    for m in get_lora_modules(model):
        m.initialize_parameters()

    return model

class FedPPORecipe(FTRecipeInterface):
    """
    Full finetuning recipe for RLHF with PPO for dense transformer-based LLMs such as LLama2. This recipe is optimized
    for single GPU training. Training on CPU is not supported.

    This implementation is based on `Learning to summarize from human feedback <https://arxiv.org/abs/2009.01325`_ and
    `Training a Helpful and Harmless Assistant with Reinforcement Learning from Human Feedback <https://arxiv.org/abs/2204.05862`_.

    Features:
        - Activation Checkpointing. This can be controlled using the ``activation_checkpointing``
            flag. Activation checkpointing helps reduce the memory footprint since we no longer keep
            activations in memory and instead recompute them during the backward pass. This is especially
            helpful for larger batch sizes when you're memory constrained. But these savings in memory
            come at the cost of training performance. In most cases training can slow-down quite a bit as
            a result of this activation recomputation.

        - Precision. Full fp32 and bf16 training are supported. Precision is controlled using the ``dtype``
            flag. When ``dtype=bf16``, all activations, gradients and optimizer states are in bfloat16. In
            most cases this should halve the memory footprint of full precision (fp32) training, without
            loss in model quality (will depend on the model, training data and other settings). For
            GPUs which do not support bfloat16, we fall back to fp32. Mixed precision training and fp16
            precision are currently not supported.

        - Adjusting batch sizes when memory constrained. This recipe uses three different batch sizes:
            - ``batch_size`` controls the total number of samples which are sampled from the dataset for a single trajectory.
            - ``forward_batch_size`` controls the mini-batch size for trajectory generation. Since gradients are disabled
                during trajectory generation, memory consumption is lower and this can be higher than ``ppo_batch_size``.
            - ``ppo_batch_size`` controls the number of samples used for a single optimization step during PPO optimization.
                Since we're optimizing two models at once, adjusting this parameter can have a big impact during training.

        - Gradient Accumulation. You can simulate larger ``ppo_batch_size`` sizes by accumulating gradients. This is
            controlled using the ``gradient_accumulation_steps`` flag.

            For example: with ``ppo_batch_size``=32 and ``gradient_accumulation_steps``=16, each backward pass during
            PPO optimization uses a 'micro batch size' of 2.

            Gradient accumulation is especially useful when you are memory constrained. In this case,
            accumulating gradients might give you better training speed than enabling activation
            checkpointing.

        - Optimizer in Backward. Fusing the optimizer step into the backward pass helps reduce the memory
            footprint associated with gradients. This can be especially helpful when you are memory
            constrained. Note that users can only use ONE of gradient accumulation or optimizer in backward.
            These features currently do not work together. For more details on optimizer in backward, please
            see this tutorial: https://pytorch.org/tutorials/intermediate/optimizer_step_in_backward_tutorial.html

            This paramater can provide significant performance gains, since there the number of optimization steps
            scales with ``ppo_epochs`` and ``batch_size``. Depending on the maximum sequence length sampled from the dataset,
            we've found that setting ``ppo_batch_size`` to the highest you can fit in memory, and `optimizer_in_bwd=True` to
            provide significant memory savings.

        - Lower precision optimizers. This recipe supports lower-precision optimizers from the bitsandbytes
            library (https://huggingface.co/docs/bitsandbytes/main/en/index). We've tested the recipe with
            8-bit AdamW and Paged AdamW. These optimizers are especially helpful when you are memory constrained
            since they help reduce the memory footprint associated with the optimizer states.

        - Checkpointing. Model weights are checkpointed only at the end of training. Resuming training is unsupported.

            For more details on the checkpointer, please take a look at
            our checkpointer deepdive (https://pytorch.org/torchtune/main/deep_dives/checkpointer.html).

        - Logging. Terminal, Disk, WandB and TensorBoard are all supported.

    Args:
        cfg (DictConfig): OmegaConf object parsed from yaml file

    Raises:
        RuntimeError: If ``dtype`` is set to fp16.
    """

    def __init__(self, cfg: DictConfig) -> None:
        """
        Initialize basic things.
        """
        # set device
        self._device = utils.get_device(device=cfg.device)
        if self._device.type != "cuda":
            raise RuntimeError("CUDA support. Only.")

        # set dtype
        self._dtype = training.get_dtype(cfg.dtype, device=self._device)
        # No necessary features s.a. gradient scaling for fp16 are implemented..
        if self._dtype == torch.float16:
            raise RuntimeError(
                "Full fp16 training is not supported. Please use bf16 or fp32 instead."
            )

        # logging attributes
        self._output_dir = cfg.output_dir
        self._log_every_n_steps = cfg.get("log_every_n_steps", 1)
        self._log_peak_memory_stats = cfg.get("log_peak_memory_stats", False)

        # manually setting up a generator for the recipe
        self.seed = training.set_seed(seed=cfg.seed)
        self._rng = torch.Generator(self._device).manual_seed(self.seed)

        # init step counters
        self.global_step    = 0
        self._total_steps   = 0
        self._steps_run     = 0
        self._total_epochs  = 0
        self._epochs_run    = 0

        # reference policy update schedule
        self._update_ref_policy_every_n_steps = cfg.get("update_ref_policy_every_n_steps", 1)

        # save adapter configs for checkpointing
        self._policy_adapter_config = get_adapter_config(cfg.policy)
        self._valmod_adpater_config = get_adapter_config(cfg.valmod)

    def setup(self, cfg: DictConfig) -> None:
        """
        Sets up the recipe state correctly.
        """
        self._metric_logger = self._setup_metric_logger(cfg.metric_logger)
        # log the final config with parameter override
        if dist.get_rank() == 0:
            self._metric_logger.log_config(cfg)

        # setup checkpointers
        self._checkpointer: Checkpointer = config.instantiate(
            cfg.checkpointer, resume_from_checkpoint=False,
        )
        self._value_checkpointer: Checkpointer = config.instantiate(
            cfg.value_checkpointer, resume_from_checkpoint=False,
        )
        # load checkpoints
        policy_state_dict = self._checkpointer.load_checkpoint()
        valmod_state_dict = self._value_checkpointer.load_checkpoint()

        # update recipe state
        # ``_setup_model`` handles initialization and loading the state dict. This method
        # should be called before ``_setup_optimizer`` since transforming the optimizer
        # state dict requires the model
        self._policy, self._valmod = self._setup_models(
            cfg_policy          = cfg.policy,
            cfg_valmod          = cfg.valmod,
            policy_state_dict   = policy_state_dict[training.MODEL_KEY],
            valmod_state_dict   = valmod_state_dict[training.MODEL_KEY],
            compile_model       = cfg.compile,
            enable_activation_checkpointing = cfg.enable_activation_checkpointing,
        )

        # setup tokenizer
        self._tokenizer = self._setup_tokenizer(cfg)

        # setup optimizer. May be none if fused in backward pass
        self._optimizer: Optional[Optimizer] = None

        if cfg.optimizer_in_bwd:
            self._optimizer = self._setup_optimizer(cfg.optimizer)
        else:
            self._setup_in_bwd_optimizer(cfg.optimzer)

        # instantiate loss
        self._loss_fn = config.instantiate(cfg.loss)

        # setup sampler and dataloader
        self._sampler, self._dataloader = self._setup_data(
            cfg_dataset = cfg.dataset,
            tokenizer   = self._tokenizer,
            shuffle     = cfg.shuffle,
            batch_size  = cfg.batch_size,
        )
        # set other parameters
        self._setup_batch_sizes(cfg)
        self._setup_hyperparameters(cfg)

    def _setup_metric_logger(
        self,
        cfg_logger: DictConfig
    ) -> MetricLoggerInterface:
        """
        Sets up metric logger for each process.
        """
        if name := cfg_logger.get("name"):
            name = f"{name}-r{dist.get_rank()}/{dist.get_world_size()}"
        logger: MetricLoggerInterface = config.instantiate(
            cfg_logger,
            name=name
        )
        return logger

    def _setup_tokenizer(self, cfg: DictConfig) -> ModelTokenizer:
        """
        Sets up tokenizer and validates all the stop token stuff.
        """
        tokenizer: ModelTokenizer = config.instantiate(cfg.tokenizer)

        # lots of hand holding for stop tokens
        if cfg.get("stop_token_ids", False):
            stop_token_ids = cfg.stop_token_ids
            if tokenizer.eos_id not in stop_token_ids:
                warn(
                    f"tokenizer eos_id ({self._tokenizer.eos_id}) is not in stop_token_ids"
                    "({stop_token_ids}). This may lead to unexpected behaviour."
                )
        else:
            if not hasattr(tokenizer.stop_tokens):
                warn(
                    "No stop tokens defined in tokenizer, and no stop_token_ids provided."
                    "This may lead to unexpected behaviour."
                )
                stop_token_ids = []
            else:
                stop_token_ids = tokenizer.stop_tokens
        self._stop_token_ids = torch.tensor(stop_token_ids, device=self._device)

        return tokenizer

    def _setup_hyperparameters(self, cfg: DictConfig) -> None:
        """
        Sets up the training hyperparameters for the recipe. This includes the GAE hyperparameters,
        generation hyperparameters, reward masking hyperparameters, and stop token ids.
        """
        # KL-penalty coefficient
        self._kl_coeff = cfg.kl_coeff

        # GAE hyperparameters
        self._gamma = cfg.gamma
        self._lmbda = cfg.lmbda
        self._whiten_rewards = cfg.whiten_rewards

        # trajectory generation args
        self._temperature = cfg.temperature
        self._top_k = cfg.top_k
        self._max_generated_tokens = cfg.max_generated_tokens

        # reward masking args
        self._min_response_length = cfg.min_response_length
        self._penalise_no_eos = cfg.penalise_no_eos
        self._reward_penalty = cfg.reward_penalty

    def _setup_batch_sizes(self, cfg: DictConfig) -> None:
        """
        Validates and sets up parameters for used during training and for tracking training state,
        batch sizes for model forward passes during trajectory generation, PPO minibatches, and
        PPO microbatches for gradient accumulation.

        Raises
            - ValueError if:
                - batch_size is not divisible by forward_batch_size
                - batch_size is not divisible by ppo_batch_size
                - ppo_batch_size is not divisible by gradient_accumulation_steps
                - num_steps is less than batch_size
                - gradient_accumulation_steps > 1 and optimizer_in_bwd is True
        """
        self.batch_size = cfg.batch_size
        self._forward_batch_size = cfg.forward_batch_size
        self._ppo_epochs = cfg.ppo_epochs
        self._ppo_batch_size = cfg.ppo_batch_size
        self._gradient_accumulation_steps = cfg.gradient_accumulation_steps
        self._ppo_backward_batch_size = (
            cfg.ppo_batch_size // self._gradient_accumulation_steps
        )

        if self.batch_size % self._forward_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"forward_batch_size ({self._forward_batch_size})."
            )
        if self.batch_size % self._ppo_batch_size != 0:
            raise ValueError(
                f"batch_size ({self.batch_size}) must be exactly divisible by "
                f"ppo_batch_size ({self._ppo_batch_size})."
            )
        if self._ppo_batch_size % self._gradient_accumulation_steps != 0:
            raise ValueError(
                f"ppo_batch_size ({self._ppo_batch_size}) must be exactly divisible "
                f"by gradient_accumulation_steps ({self._gradient_accumulation_steps})."
            )

        if self._gradient_accumulation_steps > 1 and self._optimizer_in_bwd:
            raise RuntimeError(
                "Gradient accumulation is not supported with optimizer in bwd."
                "Please set gradient_accumulation_steps=1, or optimizer_in_bwd=False."
            )

        self._total_steps = cfg.num_steps // self.batch_size
        batches_per_epoch = max(
            1, len(self._dataloader)
        )  # when we only have a single batch in the dataset

        self._total_epochs = math.ceil(self._total_steps / batches_per_epoch)
        if self._total_steps == 0:
            raise ValueError(
                f"num_steps {cfg.num_steps} must be greater than the batch size {self.batch_size}."
            )
        if self._total_steps < len(self._dataloader):
            warn(
                f"There are fewer total steps ({self._total_steps}, (num_steps//batch_size) "
                f"than there are batches ({len(self._dataloader)}) in the dataset. "
                f"Training will stop after ({self._total_steps}) steps without saving intermediate checkpoints"
            )
        if (self._total_steps > batches_per_epoch) and (
            self._total_steps % batches_per_epoch != 0
        ):
            warn(
                f"num_steps ({cfg.num_steps}) is not exactly divisible by "
                f"the number of batches in the dataset ({batches_per_epoch}). "
                f"Intermediate checkpoints will only be saved every {batches_per_epoch} steps."
            )
        log_rank_zero(
            log, f"Total steps to run: {self._total_steps}, Total epochs to run: {self._total_epochs}"
        )

    def _setup_models(
        self,
        cfg_policy: DictConfig,
        cfg_valmod: DictConfig,
        policy_state_dict: Dict[str, Any],
        valmod_state_dict: Dict[str, Any],
        compile_model: bool,
        enable_activation_checkpointing: bool,
    ) -> Tuple[nn.Module, nn.Module, nn.Module]:
        """
        Sets up the policy/reference model and reward/value model.
        """
        with training.set_default_dtype(self._dtype), self._device:
            policy: nn.Module = config.instantiate(cfg_policy)
            valmod: nn.Module = config.instantiate(cfg_valmod)

            set_trainable_params(policy, get_adapter_params(policy))
            set_trainable_params(valmod, get_adapter_params(valmod))

        if enable_activation_checkpointing:
            training.set_activation_checkpointing(
                policy, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )
            training.set_activation_checkpointing(
                valmod, auto_wrap_policy={modules.TransformerSelfAttentionLayer}
            )

        # warning: strict=False mode allows undesirable state dict - model parameters mismatch that
        # may result in incorrect initialization but is set to load with no LoRAs predefined.
        policy.load_state_dict(policy_state_dict, strict=False)

        # since we should be loading a classifier checkpoint into
        # a classifier model, this function should just ensure
        # output.weight appears in the state_dict and the model's parameters,
        # and removes output.bias from the state dict if found
        training.update_state_dict_for_classifier(
            valmod_state_dict, valmod.named_parameters()
        )
        # warning: strict=False mode allows undesirable state dict - model parameters mismatch that
        # may result in incorrect initialization but is set to load with no LoRAs predefined.
        valmod.load_state_dict(valmod_state_dict, strict=False)

        # Validate models were loaded in with the expected dtype.
        training.validate_expected_param_dtype(
            policy.named_parameters(), dtype=self._dtype
        )
        training.validate_expected_param_dtype(
            valmod.named_parameters(), dtype=self._dtype
        )
        log_rank_zero(
            log, f"Models are initialized with {self._dtype} precision."
        )

        # Compile model, if enabled.
        if compile_model:
            backend = os.environ.get("TORCH_COMPILE_BACKEND", "inductor")
            log_rank_zero(
                log, "NOTE: torch.compile is enabled and model would be compiled in first forward."
                "Expect a relatively slow first iteration."
            )
            policy.compile(backend=backend)
            valmod.compile(backend=backend)

        if dist.get_rank() == 0:
            memory_stats = training.get_memory_stats(device=self._device)
            training.log_memory_stats(memory_stats)

        return policy, valmod

    def _setup_optimizer(self, cfg_optimizer: DictConfig) -> Optimizer:
        """
        Regular optimizer setup.
        """
        optimizer = config.instantiate(
            cfg_optimizer,
            chain(self._policy.parameters(), self._valmod.parameters()),
        )
        log_rank_zero(log, "Optimizer is set up.")
        return optimizer

    def _setup_in_bwd_optimizer(self, cfg_optimizer: DictConfig) -> None:
        """
        In-backward optimizer setup.
        """
        # Maintain a dict of optims for every parameter.
        optim_dict = {
            p: config.instantiate(cfg_optimizer, [p])
            for p in chain(
                self._policy.parameters(), self._valmod.parameters()
            )
        }
        # Register optimizer step hooks on the models to run optimizer in backward.
        training.register_optim_in_bwd_hooks(
            model=self._policy, optim_dict=optim_dict
        )
        training.register_optim_in_bwd_hooks(
            model=self._valmod, optim_dict=optim_dict
        )
        log_rank_zero(log, "In-backward optimizers are set up.")

    def _setup_data(
        self,
        cfg_dataset: DictConfig,
        tokenizer: ModelTokenizer,
        batch_size: int,
        shuffle: bool,
    ) -> Tuple[DistributedSampler, DataLoader]:
        """
        All data related setup happens here.
        """
        if isinstance(cfg_dataset, ListConfig):
            datasets = [
                config.instantiate(single_cfg_dataset, tokenizer=tokenizer)
                for single_cfg_dataset in cfg_dataset
            ]
            ds = ConcatDataset(datasets=datasets)
        else:
            ds = config.instantiate(cfg_dataset, tokenizer=tokenizer)

        sampler = DistributedSampler(ds, shuffle=shuffle)
        dataloader = DataLoader(
            dataset=ds,
            sampler=sampler,
            batch_size=batch_size,
            # dropping last avoids shape issues with compile + flex attention
            drop_last=True,
            collate_fn=partial(
                padded_collate,
                pad_direction="left",
                keys_to_pad=["tokens", "labels"],
                padding_idx=tokenizer.pad_id,
            ),
        )
        return sampler, dataloader

    def save_checkpoint(self, epoch: int) -> None:
        """
        Save state dict to file. Policy and value models are merged with their
        adapters and saved. Adapters with corresponding configs are saved
        separately as well.
        """
        self._checkpointer.save_checkpoint(
            state_dict=get_merged_adapter_state_dict(
                module=self._policy,
                module_adapter_config=self._policy_adapter_config
            ),
            epoch=epoch
        )
        self._value_checkpointer.save_checkpoint(
            state_dict=get_merged_adapter_state_dict(
                module=self._valmod,
                module_adapter_config=self._valmod_adpater_config
            ),
            epoch=epoch
        )

    def generate_trajectory(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generates a trajectory given the current policy and value models, the reference policy model, the reward model,
        and batch of inputs. This is done over the following steps:

        1: Generate responses, and logits corresponding to the responses using the current policy,
            generating (query, response) pairs.
        2. Estimate logprobs of the generated responses using the current policy.
        3. Estimate values from the generated responses using the current value function.
        4. Replace any tokens in the response after the first stop token (usually EOS token) with padding,
            producting truncated responses.
        5. Run the reward model on the (query, truncated-response) pairs.
        6. Mask out all the invalid values in the trajectory due to padding tokens.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.Trajectory` comprising
                the current trajectory.
        """
        batch_size, context_length = input_ids.shape

        # step 1: generate responses, and logits corresponding to the responses using the current policy
        query_responses, logits = generation.generate(
            model=self._policy,
            prompt=input_ids,
            max_generated_tokens=self._max_generated_tokens,
            temperature=self._temperature,
            top_k=self._top_k,
            pad_id=self._tokenizer.pad_id,
            rng=self._rng,
        )

        responses = query_responses[:, context_length:].clone()
        query_response_padding_masks = query_responses != self._tokenizer.pad_id

        # step 1.1 create attention masks and position IDs for any padding tokens in inputs, used for future forward passes
        masks = generation.get_causal_mask_from_padding_mask(
            query_response_padding_masks
        )
        position_ids = generation.get_position_ids_from_padding_mask(
            query_response_padding_masks
        )

        del query_response_padding_masks

        # step 2. estimate logprobs of the responses using the current policy
        logits = logits[:, context_length - 1 :]
        logprobs = rlhf.logits_to_logprobs(logits, responses, self._temperature)

        del logits

        # step 2.1 estimate logprobs of the responses using the reference policy
        with torch.no_grad(), disable_adapter(self._policy):
            ref_logits = self._policy(
                query_responses, input_pos=position_ids, mask=masks
            )
        ref_logits = rlhf.truncate_sequence_for_logprobs(ref_logits, context_length)
        ref_logprobs = rlhf.logits_to_logprobs(ref_logits, responses, self._temperature)

        del ref_logits

        # step 3. estimate values from the responses using the value function
        values = self._valmod(query_responses, input_pos=position_ids, mask=masks)
        values = rlhf.truncate_sequence_for_logprobs(values, context_length).squeeze(-1)

        # step 4. replace any tokens in the responses after the first stop token (usually EOS token) with padding
        # resulting in truncated responses
        response_padding_masks, responses = rlhf.truncate_sequence_at_first_stop_token(
            responses, self._stop_token_ids, self._tokenizer.pad_id
        )

        # step 5. run the reward model on the (query, truncated-response) pairs
        with torch.no_grad(), disable_adapter(self._valmod):
            scores = self._valmod(
                torch.cat([input_ids, responses], dim=1),
                input_pos=position_ids,
                mask=masks,
            )

        del responses

        # step 5.1 the scores from the reward model are the logits for the last non-padding token in
        # each (query, truncated-response) pair
        seq_lens = training.get_unmasked_sequence_lengths(response_padding_masks)
        scores = scores[torch.arange(batch_size), seq_lens + context_length].squeeze(-1)

        # step 5.2 if configured, apply any penalties for sequences without EOS tokens
        # or shorter than a certain length
        if self._penalise_no_eos or self._min_response_length:
            reward_penalty_mask = rlhf.get_reward_penalty_mask(
                response_padding_masks,
                seq_lens,
                self._penalise_no_eos,
                self._min_response_length,
            )
            scores[reward_penalty_mask] = self._reward_penalty

        # step 6. mask out all the invalid values in the trajectory due to padding tokens
        logprobs[response_padding_masks] = 1.0
        ref_logprobs[response_padding_masks] = 1.0

        # step 6.1 values are masked out *after* the last valid token in the response
        value_seq_idxs = torch.where(
            (seq_lens > 0) & (seq_lens < self._max_generated_tokens - 1),
            seq_lens + 1,
            seq_lens,
        )
        value_padding_masks = response_padding_masks.clone()
        value_padding_masks[
            torch.arange(batch_size, device=value_padding_masks.device),
            value_seq_idxs,
        ] = False

        values[value_padding_masks] = 0.0

        return Trajectory(
            query_responses=query_responses,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs,
            values=values,
            masks=masks,
            position_ids=position_ids,
            response_padding_masks=response_padding_masks,
            value_padding_masks=value_padding_masks,
            value_seq_idxs=value_seq_idxs,
            scores=scores,
            seq_lens=seq_lens,
        )

    def generate_trajectory_batched(self, input_ids: torch.Tensor) -> Trajectory:
        """
        Generates a ``self.batch_size`` batch of trajectories using `self._forward_batch_size` batch sizes.
        See ``generate_trajectory`` for more details.

        Args:
            input_ids (torch.Tensor): tensor of input token IDs with shape [b, seq_length]

        Returns:
            Trajectory: An instance of :class:`~torchtune.rlhf.Trajectory`, comprising
                the current trajectory.
        """
        trajectories: List[Trajectory] = []
        with torch.no_grad():
            for batch_start in range(0, self.batch_size, self._forward_batch_size):
                batch_input_ids = input_ids[
                    batch_start : batch_start + self._forward_batch_size
                ]
                trajectories.append(self.generate_trajectory(batch_input_ids))
        return Trajectory(*map(torch.cat, zip(*trajectories)))

    def train(self) -> None:
        """
        The core training loop.
        """
        # zero out the gradients before starting training
        if self._optimizer is not None:
            self._optimizer.zero_grad()

        training_completed = False
        pbar = tqdm(total=self._total_steps, initial=self._steps_run)
        for curr_epoch in range(self._epochs_run, self._total_epochs):
            # Ensure data is not reshuffled at new epoch so the agents are
            # trained on non-overlapping data.
            self._sampler.set_epoch(0)

            for _, batch in enumerate(self._dataloader):
                batch = batch["tokens"].to(self._device)
                _, context_length = batch.shape

                # step 1. generate the trajectory using:
                # - the current policy (pi_theta)
                # - the current value function (V_phi)
                # - the reference frozen policy model (pi_theta_0)
                trajectory = self.generate_trajectory_batched(batch)

                # step 2. get the rewards for the current trajectory. these are based on:
                #   - the divergence between the current policy and the reference policy
                #   - the scores from the reward model
                rewards, kl, kl_rewards = rlhf.get_rewards_ppo(
                    trajectory.scores,
                    trajectory.logprobs,
                    trajectory.ref_logprobs,
                    self._kl_coeff,
                    trajectory.value_seq_idxs,
                )

                # step 3. estimate the advantages using Generalized Advantage Estimation (GAE)
                advantages, returns = rlhf.estimate_advantages(
                    trajectory.values,
                    rewards,
                    self._gamma,
                    self._lmbda,
                    masks=~trajectory.response_padding_masks,
                )

                # step 4. optimise using the PPO objective over multiple epochs
                ppo_stats: List[PPOStats] = []
                for _ in range(self._ppo_epochs):
                    batch_idxs = torch.randperm(self.batch_size, device=self._device)
                    for i in range(0, self.batch_size, self._ppo_batch_size):
                        mini_batch_idxs = batch_idxs[i : i + self._ppo_batch_size]

                        batch_ppo_stats: List[PPOStats] = []
                        for j in range(
                            0, self._ppo_batch_size, self._ppo_backward_batch_size
                        ):
                            backward_batch_idxs = mini_batch_idxs[
                                j : j + self._ppo_backward_batch_size
                            ]

                            batch_trajectory = Trajectory(
                                *map(
                                    partial(
                                        torch.index_select,
                                        dim=0,
                                        index=backward_batch_idxs,
                                    ),
                                    trajectory,
                                )
                            )
                            batch_ppo_stats.append(
                                self._ppo_step(
                                    batch_trajectory,
                                    advantages[backward_batch_idxs],
                                    returns[backward_batch_idxs],
                                    context_length,
                                )
                            )
                            del batch_trajectory

                        ppo_stats.append(PPOStats(*map(sum, zip(*batch_ppo_stats))))

                        grad_logs = {
                            **self._collect_grad_norm("policy", self._policy),
                            **self._collect_grad_norm("value", self._valmod)
                        }

                        if self._optimizer is not None:
                            self._optimizer.step()
                            self._optimizer.zero_grad(set_to_none=True)

                        self.global_step += 1

                # step 5. profit
                self._steps_run += 1
                if self._steps_run % self._log_every_n_steps == 0:
                    self.log_metrics(
                        trajectory,
                        PPOStats(*map(torch.stack, zip(*ppo_stats))),
                        kl,
                        kl_rewards,
                        **grad_logs
                    )
                self.cleanup_after_step(
                    trajectory, ppo_stats, advantages, returns, kl, kl_rewards
                )
                if self._steps_run % self._update_ref_policy_every_n_steps == 0:
                    # effectively update reference policy.
                    merge_lora_adapter(self._policy)
                    clear_lora_adapter(self._policy)

                pbar.update(1)
                if self._steps_run == self._total_steps:
                    training_completed = True
                    break

            self._epochs_run += 1

            if training_completed:
                if dist.get_rank() == 0:
                    self.save_checkpoint(curr_epoch)
                return

    def _ppo_step(
        self,
        trajectory: Trajectory,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        context_length: int,
    ) -> PPOStats:
        """
        Perform a single PPO optimisation step over a batch of trajectories and corresponding advantages and returns.

        Args:
            trajectory (Trajectory): a batch of trajectories
            advantages (torch.Tensor): advantages corresponding to the trajectories
            returns (torch.Tensor): returns corresponding the trajectories
            context_length (int): input ids sequence length

        Returns:
            PPOStats: An instance of :class:`~torchtune.rlhf.PPOStats`, a NamedTuple containing:
               - loss (torch.Tensor): The total PPO loss.
               - policy_loss (torch.Tensor): The policy function loss.
               - value_loss (torch.Tensor): The value function loss.
               - ratios (torch.Tensor): The ratio between the current and old policy probabilities.
               - clipfrac (torch.Tensor): The fraction of ratios that were clipped.
               - approx_policy_kls: Average estimated KL divergence between the policy before and after the optimisation step.

        """
        # estimate logprobs from the policy at the current optimisation step
        pi_logits = self._policy(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )
        pi_logits = rlhf.truncate_sequence_for_logprobs(pi_logits, context_length)
        pi_logprobs = rlhf.logits_to_logprobs(
            pi_logits, trajectory.query_responses[:, context_length:], self._temperature
        )
        pi_logprobs[trajectory.response_padding_masks] = 1.0

        del pi_logits

        # estimate the values from the value function at the current optimisation step
        phi_values = self._valmod(
            trajectory.query_responses,
            input_pos=trajectory.position_ids,
            mask=trajectory.masks,
        )

        phi_values = rlhf.truncate_sequence_for_logprobs(
            phi_values, context_length
        ).squeeze(-1)
        phi_values[trajectory.value_padding_masks] = 0.0

        # calculate ppo loss
        loss, policy_loss, value_loss, ratios, clipfrac = self._loss_fn(
            trajectory.logprobs,
            pi_logprobs,
            advantages,
            trajectory.values,
            phi_values,
            returns,
            padding_masks=~trajectory.response_padding_masks,
            value_padding_masks=~trajectory.value_padding_masks,
        )

        loss /= self._gradient_accumulation_steps
        loss.backward()

        with torch.no_grad():
            approx_policy_kls = (
                0.5 * (pi_logprobs - trajectory.logprobs).pow(2)
            ).mean()

        return PPOStats(
            loss,
            policy_loss / self._gradient_accumulation_steps,
            value_loss / self._gradient_accumulation_steps,
            ratios / self._gradient_accumulation_steps,
            clipfrac / self._gradient_accumulation_steps,
            approx_policy_kls / self._gradient_accumulation_steps,
        )

    def _collect_grad_norm(self, name: str, module: nn.Module) -> dict[str, torch.Tensor]:
        return {
            f"{name}_lora_grad_norm":
                grad_norm([param for name, param in module.named_parameters() if "lora" in name]),
            f"{name}_base_grad_norm":
                grad_norm([param for name, param in module.named_parameters() if "lora" not in name]),
        }

    def log_metrics(
        self,
        trajectory: Trajectory,
        ppo_stats: PPOStats,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
        **kwargs
    ) -> None:
        """
        Log metrics and statistics for the current step to the metric logger.
        """
        log_dict = {
            "scores": trajectory.scores.mean(),
            "num_stop_tokens": trajectory.response_padding_masks.any(-1).sum(),
            "rlhf_reward": trajectory.scores.mean() + kl_rewards.sum(1).mean(),
            "kl": kl.sum(1).mean(),
            "kl_reward": kl_rewards.sum(1).mean(),
            "loss": ppo_stats.loss.mean(),
            "policy_loss": ppo_stats.policy_loss.mean(),
            "value_loss": ppo_stats.value_loss.mean(),
            "clipfrac": ppo_stats.clipfrac.mean(),
            "ratios": ppo_stats.ratios.mean(),
            "approx_policy_kl": ppo_stats.approx_policy_kls.mean(),
            "response_lengths": trajectory.seq_lens.float().mean(),
            **kwargs
        }
        if self._device.type == "cuda" and self._log_peak_memory_stats:
            log_dict.update(training.get_memory_stats(device=self._device))

        self._metric_logger.log_dict(log_dict, step=self.global_step)

    def cleanup_after_step(
        self,
        trajectory: Trajectory,
        ppo_stats: PPOStats,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        kl: torch.Tensor,
        kl_rewards: torch.Tensor,
    ) -> None:
        """
        Cleanup tensors after each PPO step to free up memory.
        """
        # there shouldn't be any floating references to the individual tensors at the this point, so gc can do its thing
        for v in trajectory:
            del v
        del trajectory
        for v in ppo_stats:
            del v
        del ppo_stats
        del advantages
        del returns
        del kl
        del kl_rewards

    def cleanup(self, **kwargs) -> None:
        self._metric_logger.close()
        dist.destroy_process_group()


@config.parse
def recipe_main(cfg: DictConfig) -> None:
    """
    Entry point for the recipe.

    Configurable parameters are read in the following order:
        - Parameters specified in config (see available configs through ``tune ls``)
        - Overwritten by arguments from the command-line
    """
    dist.init_process_group()
    config.log_config(recipe_name="FedPPORecipe", cfg=cfg)
    recipe = FedPPORecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.train()
    recipe.cleanup()


if __name__ == "__main__":
    sys.exit(recipe_main())
