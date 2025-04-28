import typing as tp
import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchtune.modules.transforms.tokenizers import ModelTokenizer
from ppotune.arbiters.pairwise_arbiter import PairwiseArbiter
from ppotune.log import WandbLogger
from ppotune.model import GenerativeModel


logger = WandbLogger()

class Evaluator(tp.Protocol):
    """
    Evaluator protocol.
    """
    def __call__(
        model: GenerativeModel,
        step: int
    ) -> None:
        """
        Performs evaluation at each n-th step and logs the result.
        """
        ...

class ReferenceCompletionEvaluator(Evaluator):
    """
    Evaluates model based on reference completion dataset w.r.t pairwise
    arbiter opinion.
    """
    def __init__(
        self,
        arbiter: PairwiseArbiter,
        tokenizer: ModelTokenizer,
        dataloader: DataLoader,
        every_n_steps: int,
    ) -> None:

        self._arbiter = arbiter
        self._tokenizer = tokenizer
        self._dataloader = dataloader
        self._every_n_steps = every_n_steps

    def __call__(
        self,
        model: GenerativeModel,
        step: int = 0,
    ) -> None:

        if step % self._every_n_steps != 0:
            return

        prompts:        tp.List[str] = []
        completions:    tp.List[tp.Tuple[str, str]] = []

        decode = lambda tokens: self._tokenizer.decode(
            tokens.tolist(), skip_special_tokens=True
        )
        for batch in tqdm(
            self._dataloader,
            desc="Evaluation",
            disable=dist.get_rank() != 0
        ):
            batch["tokens"] = batch["tokens"].to(model._device)
            generated = model.generate(prompt=batch["tokens"])

            queries = []
            responses = []
            for tokens, query_mask, response_mask in zip(
                generated.tokens, generated.query_mask, generated.response_mask
            ):
                queries.append(decode(tokens[query_mask]))
                responses.append(decode(tokens[response_mask]))

            prompts.extend(queries)
            completions.extend(list(zip(batch["completion"], responses)))

        wins = torch.tensor(self._arbiter.judge(prompts, completions))
        valid = wins != -1
        winrate = wins[valid].float().mean()
        logger.collect("winrate", winrate)


def reference_completion_evaluator(
        arbiter: PairwiseArbiter,
        tokenizer: ModelTokenizer,
        dataloader: DataLoader,
        every_n_steps: int,
) -> ReferenceCompletionEvaluator:

    return ReferenceCompletionEvaluator(
        arbiter=arbiter,
        tokenizer=tokenizer,
        dataloader=dataloader,
        every_n_steps=every_n_steps,
    )
