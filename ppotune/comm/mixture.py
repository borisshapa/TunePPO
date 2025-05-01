import torch

from torchtune.modules import TransformerDecoder
from ppotune.comm.primitives import all_gather_uneven, all_to_all
from ppotune.comm.protocols import CommProtocol
from ppotune.data.types import PPOTrajectoryStats
from ppotune.log import WandbLogger
from ppotune.peft import merge_lora_adapter, clear_lora_adapter


log = WandbLogger()
# ----------------------- Distributed Policy Mixture ------------------------ #
#
class DistributedPolicyMixture:
    """
    Implements interprocess communication and delivers a mixtxture of policies.
    """
    def __init__(
        self,
        protocol: CommProtocol,
        local_policy: TransformerDecoder,
        update_every_n_steps: int,
    ) -> None:
        self._protocol = protocol
        self._policy = local_policy
        self._update_every_n_steps = update_every_n_steps

    def forward(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Follows the interface of TransformerDecoder forward method. Use it as a
        more comprehensive reference.
        Args:
            tokens (torch.Tensor): input tensor with shape ``[b x s]``
            mask (torch.Tensor): used to mask the scores after the query-key
                multiplication and before the softmax. This parameter is
                required during inference.
            input_pos (torch.Tensor): contains the position ids of each token.
                During training, this is used to indicate the positions of each
                token relative to its sample when packed, shape ``[b x s]``.
                During inference, this indicates the position of the current
                token. This parameter is required during inference.

        Returns:
            torch.Tensor: output tensor with shape ``[b x s x v]``

        Shape notation:
            - b: batch size
            - s: token sequence length
            - v: vocab size
        """
        peer_tokens = all_gather_uneven(tokens)
        peer_masks = all_gather_uneven(mask)
        peer_pos = all_gather_uneven(input_pos)

        peer_logits_requested = [
            self._policy(tokens, input_pos=pos, mask=mask)
            for tokens, pos, mask in zip(peer_tokens, peer_pos, peer_masks)
        ]
        peer_logits_responded = all_to_all(peer_logits_requested)
        return self._protocol(peer_logits_responded)

    def gather(
        self,
        stats: PPOTrajectoryStats
    ) -> None:
        """
        Gather statistics.
        """
        self._protocol.gather(stats)

    def update(
        self,
        step: int,
    ) -> None:
        """
        Update policy and communication protocol.
        """
        if step % self._update_every_n_steps != 0:
            return

        merge_lora_adapter(self._policy)
        clear_lora_adapter(self._policy)
        self._protocol.update()

    def __call__(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        input_pos: torch.Tensor,
    ) -> torch.Tensor:
        """
        Executes forward pass.
        """
        return self.forward(tokens, mask, input_pos)


# ------------------- Distributed Policy Mixture Builder -------------------- #
#
def distributed_policy_mixture(
    protocol: CommProtocol,
    local_policy: TransformerDecoder,
    update_every_n_steps: int,
) -> DistributedPolicyMixture:
    """
    Builds DistributedPolicyMixture instance.
    """
    return DistributedPolicyMixture(
        protocol,
        local_policy,
        update_every_n_steps
    )
