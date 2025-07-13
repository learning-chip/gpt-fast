"""
MODEL_PATH=/scratch/model_weights/Llama-2-7b-chat-hf/model.pth
python minimum_eval.py --checkpoint_path $MODEL_PATH \
    --tasks mmlu_high_school_computer_science mmlu_college_biology

python minimum_eval.py --checkpoint_path $MODEL_PATH \
    --tasks gpqa_diamond_zeroshot
"""

import sys
import time
from pathlib import Path
from typing import Optional

import torch

import lm_eval
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import get_task_dict
from lm_eval.evaluator import evaluate
from lm_eval.utils import make_table

from tokenizer import get_tokenizer
from model import Transformer
from generate import _load_model, encode_tokens, model_forward, device_sync
from eval import setup_cache_padded_seq_input_pos_max_seq_length_for_prefill


class GPTFastEvalWrapper(HFLM):
    """
    A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
    """
    def __init__(
        self,
        model: Transformer,
        tokenizer,
        max_seq_length: Optional[int]=None,
    ):
        # NOTE: in lm-eval>=0.4.3 (<=0.4.9 at time of writing), `pretrained` becomes required arg for `HFLM.__init__()`
        # Here still use "gpt2" default to bypass errors, then overwrite with our custom model and tokenizer
        # For reference, compare the following commits:
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.2/lm_eval/models/huggingface.py#L80
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.3/lm_eval/models/huggingface.py#L82
        # https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/models/huggingface.py#L60
        super().__init__(pretrained="gpt2")
        self._model = model
        self._tokenizer = tokenizer
        self._device = torch.device('cuda')  # TODO: correctly set devices for TP>=2 cases
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, **kwargs):
        encoded = encode_tokens(self._tokenizer,
            string, bos=True, device=self._device)
        # encoded is a pytorch tensor, but some internal logic in the
        # eval harness expects it to be a list instead
        # TODO: verify this for multi-batch as well
        encoded = encoded.tolist()
        return encoded

    def tok_decode(self, tokens):
        decoded = self._tokenizer.decode(tokens)
        return decoded

    def _model_call(self, inps):
        # TODO: make batches work
        inps = inps.squeeze(0)

        max_new_tokens = 1
        seq, input_pos, max_seq_length = \
            setup_cache_padded_seq_input_pos_max_seq_length_for_prefill(
                self._model,
                inps,
                max_new_tokens,
                self.max_length,
            )
        x = seq.index_select(0, input_pos).view(1, -1)
        logits = model_forward(self._model, x, input_pos)
        return logits

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception('unimplemented')


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
