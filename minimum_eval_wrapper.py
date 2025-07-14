# streamlined from https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9/lm_eval/models/huggingface.py

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import transformers
from tqdm import tqdm

from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import TemplateLM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import (
    Collator,
    pad_and_concat,
    stop_sequences_criteria,
)

from model import Transformer
from generate import encode_tokens, model_forward
from eval import setup_cache_padded_seq_input_pos_max_seq_length_for_prefill

eval_logger = logging.getLogger(__name__)


@register_model("minimum")
class GPTFastEvalWrapper(TemplateLM):
    _DEFAULT_MAX_LENGTH = 2048

    """
    A wrapper class for GPTFast, providing integration with the lm-evaluation-harness library.
    Minimum causal LM for easy modification.
    """
    def __init__(
        self,
        model: Transformer,
        tokenizer,
        max_seq_length: Optional[int]=None,
    ):
        super().__init__()
        device = torch.device('cuda')  # TODO: correctly set devices for TP>=2 cases
        self._max_seq_length = 2048 if max_seq_length is None else max_seq_length

        self._model = model
        self.tokenizer = self._tokenizer = tokenizer
        self._device = device

        # NOTE: assume causal decoding-only model
        self.backend = "causal"
        self.logits_cache = True

        self._max_length = max_seq_length
        self.softmax_dtype = None  # TODO: use float32 to get higher acc?

        # TODO: fix for TP case
        self._rank = 0
        self._world_size = 1

        self.custom_prefix_token_id = None

    @property
    def config(self):
        # return the associated transformers.AutoConfig for the given pretrained model.
        return self._config

    @property
    def model(self):
        return self._model

    @property
    def eot_token_id(self):
        return self._tokenizer.eos_id()

    @property
    def prefix_token_id(self):
        # it is used as prefix for loglikelihood
        if self.custom_prefix_token_id is not None:
            return self.custom_prefix_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_seq_length

    @property
    def max_gen_toks(self):
        return 50

    @property
    def batch_size(self):
        return 1   # NOTE: assume bs=1 for now

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

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
        """
        :param inps: torch.Tensor
            A torch tensor of shape [batch, (sequence_ctx + sequence_cont)] or of shape
            [batch, sequence_ctx]. the size of sequence may vary from call to call
        :return
            A torch tensor of shape [batch, sequence, vocab] with the
        logits returned from the model's decoder
        """
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

    def _select_cont_toks(
        self, logits: torch.Tensor, contlen: int = None, inplen: int = None
    ) -> torch.Tensor:
        assert contlen and inplen, (
            "Must pass input len and cont. len to select scored logits for causal LM"
        )
        # discard right-padding.
        # also discard the input/context tokens. we'll only score continuations.
        logits = logits[inplen - contlen : inplen]
        return logits

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], List[int], List[int]]],
        disable_tqdm: bool = False,
        sort_by_longest: bool = True,
        override_bs: int = None,
    ) -> List[Tuple[float, bool]]:
        # Simplified for batch_size=1: no Collator, just loop over requests
        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running loglikelihood requests",
        )

        # Track original indices.
        indexed_requests = list(enumerate(requests))

        if sort_by_longest:
            # Sort requests by descending length to catch OOM early,
            # but recover the original order in the returned results.
            # Verified that output accuracy score is not affected.
            indexed_requests = sorted(indexed_requests, key=lambda r: len(r[1][1]), reverse=True)

        res = [None] * len(requests)  # Prepare a placeholder for results in original order
        for orig_idx, request in indexed_requests:
            request_str, context_enc, continuation_enc = request
            # sanity check
            assert len(context_enc) > 0
            assert len(continuation_enc) > 0
            assert len(continuation_enc) <= self.max_length

            total_length = len(context_enc) + len(continuation_enc)
            if total_length > self.max_length + 1:
                eval_logger.warning(
                    f"Combined length of context ({len(context_enc)}) and continuation ({len(continuation_enc)}) "
                    f"exceeds model's maximum length ({self.max_length}). "
                    f"Truncating {total_length - self.max_length + 1} tokens from the left."
                )
            inp = torch.tensor(
                (context_enc + continuation_enc)[-(self.max_length + 1):][:-1],
                dtype=torch.long,
                device=self.device,
            )
            inplen = inp.shape[0]
            cont_toks = continuation_enc
            contlen = len(cont_toks)

            # Pad to batch (batch_size=1)
            batched_inps = pad_and_concat(inplen, [inp], padding_side="right")  # [1, inplen]
            multi_logits = F.log_softmax(
                self._model_call(batched_inps),
                dim=-1,
                dtype=self.softmax_dtype,
            )  # [1, inplen, vocab]
            logits = multi_logits[0]  # [inplen, vocab]

            # For batch_size=1, padding_len_inp = inplen, so ctx_len = inplen
            ctx_len = inplen
            logits = self._select_cont_toks(logits, contlen=contlen, inplen=ctx_len)
            logits = logits.unsqueeze(0)  # [1, seq, vocab]

            greedy_tokens = logits.argmax(dim=-1)
            cont_toks_tensor = torch.tensor(
                cont_toks, dtype=torch.long, device=self.device
            ).unsqueeze(0)  # [1, seq]
            max_equal = (
                greedy_tokens[:, -cont_toks_tensor.shape[1]:] == cont_toks_tensor
            ).all()

            logits_gathered = torch.gather(
                logits, 2, cont_toks_tensor.unsqueeze(-1)
            ).squeeze(-1)  # [1, seq]

            answer = (float(logits_gathered.sum()), bool(max_equal))
            res[orig_idx] = answer

            if request_str is not None:
                self.cache_hook.add_partial(
                    "loglikelihood", request_str, answer
                )
            pbar.update(1)

        pbar.close()
        return res

    def apply_chat_template(
        self, chat_history: List[Dict[str, str]], add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )
        return chat_templated

    # NOTE: below implemented functions are not needed for multiple choice "loglikelihood" tasks
    def loglikelihood_rolling(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[float]:
        raise Exception('unimplemented')

    def _model_generate(self, context, max_length, eos_token_id):
        raise Exception('unimplemented')

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        raise Exception('unimplemented')
