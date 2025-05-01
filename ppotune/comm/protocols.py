import typing as tp
import torch
import torch.distributed as dist

from abc import ABC, abstractmethod
from ppotune.comm.weightage import Weightage
from ppotune.data.types import PPOTrajectoryStats
from ppotune.log import WandbLogger


log = WandbLogger()
# ------------------------ Communication Protocols -------------------------- #
#
class CommProtocol(ABC):
    """
    Communication Protocol reduces basically to peer tensors aggregation
    strategy driven by statisics gathered throughout training.
    """
    def __init__(
        self,
        weightage: Weightage
    ) -> None:
       self._weightage = weightage
       self._weights = self._weightage()

    @abstractmethod
    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Gather statistics.
        """
        ...

    @abstractmethod
    def update(self) -> None:
        """
        Updates weights based on stats gathered and clears stats.
        """
        ...

    def __call__(
        self,
        tensors: tp.Iterable[torch.Tensor],
    ) -> torch.Tensor:
        """
        Reduces tensor list according to protocol weightage.
        """
        tensors = torch.stack(tensors)
        weights = self._weights.to(tensors[0].device)
        log.collect("self_preference", weights[dist.get_rank()])

        for _ in range(tensors.dim() - 1):
            weights = weights.unsqueeze(-1)
        return (weights * tensors).sum(dim=0)


class StaticProtocol(CommProtocol):
    """
    Executes static peer tensor reduction non-dependant on learning dynamics.
    """
    def __init__(
        self,
        weightage: Weightage,
    ) -> None:
        super().__init__(weightage)

    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Ignores stats.
        """
        pass

    def update(self) -> None:
        """
        Clears stats gathered.
        """
        self._weights = self._weightage()


class ScoreBasedProtocol(CommProtocol):
    """
    Relies on scores obtained between communication rounds.
    """
    def __init__(
        self,
        weightage: Weightage,
    ) -> None:
        super().__init__(weightage)
        self._score = 0.0

    def gather(self, stats: PPOTrajectoryStats) -> None:
        """
        Gathers mean scores.
        """
        self._score += stats.scores.mean()

    def update(self) -> None:
        """
        Clears mean scores.
        """
        self._weights = self._weightage(self._score)
        self._score = 0.0


# -------------------- Communication Protocol Builders ---------------------- #
#
def static_protocol(weightage: Weightage) -> StaticProtocol:
    return StaticProtocol(weightage)

def score_based_protocol(weightage: Weightage) -> ScoreBasedProtocol:
    return ScoreBasedProtocol(weightage)
