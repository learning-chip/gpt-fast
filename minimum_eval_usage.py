"""
MODEL_PATH=/scratch/model_weights/Llama-2-7b-chat-hf/model.pth
python minimum_eval_usage.py --checkpoint_path $MODEL_PATH \
    --tasks mmlu_high_school_computer_science mmlu_college_biology

python minimum_eval_usage.py --checkpoint_path $MODEL_PATH \
    --tasks gpqa_diamond_zeroshot

python minimum_eval_usage.py --checkpoint_path $MODEL_PATH \
    --tasks mmlu | tee run_minimumeval_mmlu.log
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch

import lm_eval
from lm_eval.tasks import get_task_dict
from lm_eval.evaluator import evaluate
from lm_eval.utils import make_table

from tokenizer import get_tokenizer
from model import Transformer
from generate import _load_model, device_sync
from minimum_eval_wrapper import GPTFastEvalWrapper


@torch.no_grad()
def eval(
    model: Transformer,
    tokenizer,
    tasks: list = ["mmlu"],
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> dict:
    """
    Evaluates a language model on a specified task using the lm-evaluation-harness library.

    Args:
        model (Transformer): The pre-trained language model to evaluate.
        tokenizer: The tokenizer to use for encoding/decoding text.
        tasks (list): The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.

    Returns:
        eval_results (dict): A dictionary of evaluation results for the specified task(s).
    """
    lm = GPTFastEvalWrapper(
        model,
        tokenizer,
        max_seq_length,
    )

    # NOTE: `initialize_tasks()` is removed since 0.4.2
    # https://github.com/EleutherAI/lm-evaluation-harness/releases/tag/v0.4.2
    task_manager = lm_eval.tasks.TaskManager()
    eval_results = lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        task_manager=task_manager
    )
    return eval_results

def main(
    checkpoint_path: Path = Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"),
    tasks: list = ["hellaswag"],
    limit: Optional[int] = None,
    max_seq_length: Optional[int] = None,
) -> None:
    """Evaluates model on a task from the `lm-evaluation-harness` library.

    Args:
        checkpoint_path (Path): The path to the model checkpoint file to load.
        tasks (list): The names of the evaluation tasks to perform.
        limit (Optional[int]): The maximum number of samples to evaluate (None for all available).
        max_seq_length (Optional[int]): The maximum sequence length allowed for input text.
    """

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), str(tokenizer_path)

    device = 'cuda'
    precision = torch.bfloat16

    # TP handling taken from `generate.py`
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None

    print(f"[rank {rank}] Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    device_sync(device=device)
    print(f"[rank {rank}] Time to load model: {time.time() - t0:.02f} seconds.")

    model.eval()

    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)

    torch.manual_seed(1234)

    t1 = time.time()
    result = eval(
        model,
        tokenizer,
        tasks,
        limit,
        max_seq_length,
    )
    print(f"[rank {rank}] Time to run eval: {time.time() - t1:.02f} seconds.")
    print(f"[rank {rank}] For model {checkpoint_path}")

    # Print results
    print(f"[rank {rank}]", make_table(result))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-llama/Llama-2-7b-chat-hf/lit_model.pth"), help='Model checkpoint path.')
    parser.add_argument('--tasks', nargs='+', type=str, default=["hellaswag"], help='list of lm-eluther tasks to evaluate usage: --tasks task1 task2')
    parser.add_argument('--limit', type=int, default=None, help='number of samples to evalulate')
    parser.add_argument('--max_seq_length', type=int, default=None, help='maximum length sequence to evaluate')

    args = parser.parse_args()
    main(
        Path(args.checkpoint_path), args.tasks, args.limit, args.max_seq_length,
    )
