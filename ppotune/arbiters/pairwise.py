import typing as tp

from abc import ABC, abstractmethod

import torch


class PairwiseArbiter(ABC):
    """
    Interface for all pairwise arbiters.
    """
    @abstractmethod
    def judge(
        self,
        prompts: tp.List[str],
        completions: tp.List[tp.Tuple[str, str]],
        shuffle_order: bool = True
    ) -> tp.List[int]:
        """
        Args:
            prompts:
                list of prompts.
            completions:
                list of tuple pairs with completions for given prompts.
            shuffle_order:
                whether to shuffle completions before passing them to the
                underlying model.
        Returns:
            list[0/1] denoting preference to 0th or 1st completion in a
                tuple respectively for each tuple in completions list.
        """
        raise NotImplementedError("Arbiters must override .judge(...) method.")


class RandomPairwiseArbiter(PairwiseArbiter):
    """
    For testting purposes.
    """
    def judge(
        self,
        prompts: tp.List[str],
        completions: tp.List[tp.Tuple[str, str]],
        shuffle_order: bool = True
    ) -> tp.List[int]:
        return torch.randint(0, 2, (len(prompts),)).tolist()


def random_pairwise_arbiter() -> RandomPairwiseArbiter:
    return RandomPairwiseArbiter()
