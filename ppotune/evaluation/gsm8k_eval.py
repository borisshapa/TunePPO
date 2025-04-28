import typing as tp

import re

import torch
import torch.distributed as dist

from tqdm import tqdm
from torch.utils.data import Dataset
from ppotune.arbiters.pairwise_arbiter import PairwiseArbiter
from ppotune.log import WandbLogger
from ppotune.model import GenerativeModel

from ppotune.evaluation.eval import ReferenceCompletionEvaluator

logger = WandbLogger()


class Gsm8kEvaluator(ReferenceCompletionEvaluator):
    def __call__(
        self,
        model: GenerativeModel,
        step: int = 0,
    ) -> None:

        if step % self._every_n_steps != 0:
            return

        prompts:     tp.List[str] = []
        completions: tp.List[tp.Tuple[str, str]] = []
        answers:     tp.List[tp.Tuple[str, str]] = []

        decode = lambda tokens: self._tokenizer.decode(
            tokens.tolist(), skip_special_tokens=True
        )
        for batch in tqdm(self._dataloader, desc="Evaluation", disable=dist.get_rank() != 0):
            batch["tokens"] = batch["tokens"].to(model._device)
            generated = model.generate(prompt=batch["tokens"])

            for tokens, query_mask, response_mask, reference_response in zip(
                generated.tokens,
                generated.query_mask,
                generated.response_mask,
                batch["answers"]
            ):
                query = decode(tokens[query_mask])
                response = decode(tokens[response_mask])

                prompt = self._extract_prompt(query)
                if prompt is None:
                    continue

                reasoning_answer = self._extract_reasoning_answer(response)
                if reasoning_answer is None:
                    continue
                reasoning, answer = reasoning_answer

                reference_reasoning_answer = self._extract_reference_reasoning_answer(
                    reference_response
                )
                if reference_reasoning_answer is None:
                    continue
                reference_reasoning, reference_answer = reference_reasoning_answer

                completion = self._make_completion(reasoning, answer)
                reference_completion = self._make_completion(reference_reasoning, reference_answer)
                
                prompts.append(prompt)
                completions.append((reference_completion, completion))
                answers.append((reference_answer, answer))

        wins = torch.tensor(self._arbiter.judge(prompts, completions))
        valid = wins != -1
        winrate = wins[valid].float().mean()
        
        answer_accuracy = torch.tensor(
            sum([ref_answer == answer for ref_answer, answer in answers]) / len(answers)
        )
        
        for idx in range(8):
            logger.collect_validation_completions(
                completions[idx][0],
                completions[idx][1],
                wins[idx]
            )
        
        logger.collect_dict({
            "winrate": winrate,
            "validation_answer_accuracy": answer_accuracy
        })
        
        return prompts, completions
    
    def _extract_prompt(self, query: str) -> str:
        match = re.search(r'User:\s*(.+?)\s*Assistant:', query, re.DOTALL)
        if match:
            return match.group(1).strip()
        
        return None

    def _extract_reasoning_answer(self, response: str) -> str:
        match = re.search(
            r'<think>(.*?)</think>\s*<answer>(.*?)</answer>',
            response,
            re.DOTALL
        )
        if match:
            reasoning = match.group(1).strip()
            answer = match.group(2).strip()
            return reasoning, answer

        return None
    
    def _extract_reference_reasoning_answer(
        self,
        reference_response: str
    ) -> str:
        lines = reference_response.strip().splitlines()
        final_line = lines[-1].strip()
        
        # Match the final answer line
        answer_match = re.match(r'^####\s*(\d+)', final_line)
        
        if answer_match:
            answer = answer_match.group(1)
            reasoning = '\n'.join(lines[:-1]).strip()
            return reasoning, answer

        return None
    
    def _make_completion(self, reasoning: str, answer: str) -> str:
        return f"Reasoning: {reasoning}\n\nAnswer: {answer}"


def gsm8k_evaluator(
        arbiter: PairwiseArbiter,
        num_samples: int,
        every_n_steps: int,
        seed: tp.Optional[int] = None,
        *, # the rest has to be defined in the recipe
        dataset: Dataset,
        batch_size: int,
) -> ReferenceCompletionEvaluator:

    return Gsm8kEvaluator(
        arbiter=arbiter,
        num_samples=num_samples,
        every_n_steps=every_n_steps,
        seed=seed,
        dataset=dataset,
        batch_size=batch_size,
    )
