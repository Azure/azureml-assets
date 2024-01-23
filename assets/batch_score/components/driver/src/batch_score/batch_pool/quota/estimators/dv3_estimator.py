# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Dv3 estimator."""

import json
import os
from abc import abstractmethod
from base64 import b64decode
from collections import Counter

import numpy as np
from tiktoken import Encoding

from ....common.telemetry import logging_utils as lu
from .quota_estimator import QuotaEstimator


class DV3Estimator(QuotaEstimator):
    DEFAULT_MAX_TOKENS = 10

    def __init__(self):
        self.__coeffs = self.__load_coeffs()
        self.__tokenizer = self.__load_tokenizer()

    def calc_tokens_with_tiktoken(self, prompt) -> "int | tuple[int]":
        if isinstance(prompt, str):
            return self.__calc_tokens_for_one_prompt(prompt)
        else:
            lu.get_logger().debug(f"Prompt is of type '{prompt.__class__.__qualname__}' "
                                  f"and length '{len(prompt)}'. Calculating each input within.")
            return self.__calc_tokens_for_batch(prompt)

    def estimate_request_cost(self, request_obj: any) -> int:
        prompt = self._get_prompt(request_obj)
        max_tokens = request_obj.get("max_tokens", self.DEFAULT_MAX_TOKENS)

        try:
            est_tokens = self.__calc_total_tokens_with_tiktoken(prompt)
        except BaseException as e:
            # This behavior can happen on some kinds of very long prompts:
            # TikToken first splits the prompt into words via regex (mostly via whitespace and
            # punctuation), then splits each word into tokens.
            # The first part is fast, but the second part seems to take O(n) stack space and O(n^2)
            # time, for n = word length. Experimentally,
            # a word above ~30k characters will be very slow, and somewhere around ~1M characters it
            # will hit a stack overflow. The problem
            # only happens when there's a single huge word -- a prompt of 1M normal-sized words works
            # fine (although DV3 itself will reject it).
            #
            # We're specifically catching BaseException here because the stack overflow error is thrown
            # from the Rust runtime as a PanicException, which inherits from BaseException, not
            # Exception, so normal `except` statements would catch it.
            #
            # See: https://github.com/openai/tiktoken/issues/15
            lu.get_logger().error(f"TikToken library call failed, falling back to estimator: {e}")
            est_tokens = self.__calc_tokens_with_linear_model(prompt)

        # Cost is total input + max output tokens.
        return est_tokens + max_tokens

    def estimate_response_cost(self, request_obj: any, response_obj: any) -> int:
        # Cost is total input + max output tokens.
        return response_obj["usage"]["prompt_tokens"] + request_obj.get("max_tokens", self.DEFAULT_MAX_TOKENS)

    def __load_coeffs(self):
        path = os.path.join(os.path.dirname(__file__), "data", "dv3_coeffs.json")

        with open(path) as f:
            return np.array(json.load(f))

    def __load_tokenizer(self):
        # Parameters taken from https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
        # We're providing these manually so we can load the tokens from a local file, instead of from a URL.
        data_path = os.path.join(os.path.dirname(__file__), "data", "cl100k_base.tiktoken")

        with open(data_path, "rb") as f:
            bpe_ranks = {b64decode(token): int(rank) for token, rank in (line.split() for line in f)}

        return Encoding(
            name="cl100k_base",
            pat_str=r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| """
                    + r"""?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
            mergeable_ranks=bpe_ranks,
            special_tokens={
                "<|endoftext|>": 100257,
                "<|fim_prefix|>": 100258,
                "<|fim_middle|>": 100259,
                "<|fim_suffix|>": 100260,
                "<|endofprompt|>": 100276,
            },
        )

    @abstractmethod
    def _get_prompt(self, request_obj: any) -> str:
        pass

    def __calc_tokens_with_linear_model(self, prompt):
        prompt_encoded = prompt.encode("utf-8")

        counts = Counter(prompt_encoded)
        histogram = np.array([counts[i] for i in range(0x100)])
        est_tokens = int(self.__coeffs.dot(histogram).round())

        # The absolute upper limit is 1 token/byte.
        return min(est_tokens, len(prompt_encoded))

    def __calc_total_tokens_with_tiktoken(self, prompt):
        count = self.calc_tokens_with_tiktoken(prompt)
        if isinstance(count, int):
            return count
        return sum(count)

    def __calc_tokens_for_one_prompt(self, prompt):
        return len(self.__tokenizer.encode(prompt, disallowed_special=()))

    def __calc_tokens_for_batch(self, input):
        return tuple(self.__calc_tokens_for_one_prompt(single_input) for single_input in input)
