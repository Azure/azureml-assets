# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deployment."""

from abc import ABC, abstractmethod
from typing import List, Any
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm
from azureml._common._error_definition.azureml_error import AzureMLError

from ...utils.constants import Constants
from ...utils.exceptions import BenchmarkValidationException
from ...utils.error_definitions import BenchmarkValidationError


class AbstractDeployment(ABC):
    """Abstract class for deployment."""

    def __init__(self):
        """Initialize Deployment."""
        self.sep = " "

    @abstractmethod
    def get_embeddings(self, text: List[str]) -> List[List[float]]:
        """
        Get embeddings for the given text.

        :param text: List of text to get embeddings for.
        :return: List of embeddings.
        """
        mssg = "`get_embeddings` method must be implemented in the derived class."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    @abstractmethod
    def get_batch_size(self, longest_sentence: str, intital_batch_size: int) -> int:
        """
        Get the batch size that will fit the model context length.

        :param longest_sentence: Longest sentence in the input text.
        :param intital_batch_size: Initial batch size.
        :return: Batch size that fits the model context length.
        """
        mssg = "`get_batch_size` method must be implemented in the derived class."
        raise BenchmarkValidationException._with_error(
            AzureMLError.create(BenchmarkValidationError, error_details=mssg)
        )

    @staticmethod
    def _text_length(text: Any):
        """Get the length for the input text."""
        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    def encode(
        self, sentences: List[str], batch_size: int = 32, **kwargs
    ) -> List[List[float]]:
        """
        Get a list of embeddings for the given list of sentences.

        :param sentences: List of sentences to encode.
        :param batch_size: Batch size for the encoding.
        :param kwargs: Additional keyword arguments.
        :return: List of embeddings for the given sentences.
        """
        all_embeddings = []

        # get the longest sentence in the input text
        longest_sentence = max(sentences, key=self._text_length)

        # reduce batch_size if the longest sentence in the input text is too long
        batch_size = self.get_batch_size(longest_sentence, batch_size)

        total_sentences = len(sentences)

        with ThreadPoolExecutor(max_workers=Constants.MAX_THREADS) as executor:
            futures = []
            with tqdm(total=total_sentences) as pbar:
                for start_index in range(0, total_sentences, batch_size):
                    sentences_batch = sentences[start_index: start_index + batch_size]
                    future = executor.submit(self.get_embeddings, sentences_batch)
                    # Update the progress bar when each future completes
                    future.add_done_callback(lambda _: pbar.update(batch_size))
                    futures.append(future)

                for future in futures:
                    embeddings = future.result()
                    all_embeddings.extend(embeddings)

        return all_embeddings
