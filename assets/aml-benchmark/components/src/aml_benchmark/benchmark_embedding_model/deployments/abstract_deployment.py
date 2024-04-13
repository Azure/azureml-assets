# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deployment."""

from abc import ABC, abstractmethod
from typing import List, Any, Dict

import numpy as np
from tqdm.autonotebook import trange
from azureml._common._error_definition.azureml_error import AzureMLError

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

        # sort sentences by length in descending order
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[idx] for idx in length_sorted_idx]

        # reduce batch_size if the longest sentence in the input text is too long
        batch_size = self.get_batch_size(sentences_sorted[0], batch_size)

        # get embeddings in batches
        for start_index in trange(
            0, len(sentences), batch_size, desc="Batches", disable=False
        ):
            sentences_batch = sentences_sorted[start_index: start_index + batch_size]
            embeddings = self.get_embeddings(sentences_batch)
            all_embeddings.extend(embeddings)

        # sort embeddings back to the original order
        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        return all_embeddings

    def encode_queries(
        self, queries: List[str], batch_size: int, **kwargs
    ) -> List[List[float]]:
        """
        Get a list of embeddings for the given list of queries.

        :param queries: List of queries to encode.
        :param batch_size: Batch size for the encoding.
        :param kwargs: Additional keyword arguments.
        :return: List of embeddings for the given queries.
        """
        return self.encode(queries, batch_size=batch_size, **kwargs)

    def encode_corpus(
        self, corpus: List[Dict[str, str]], batch_size: int, **kwargs
    ) -> List[List[float]]:
        """
        Get a list of embeddings for the given list of corpus.

        :param corpus: List of corpus to encode.
        :param batch_size: Batch size for the encoding.
        :param kwargs: Additional keyword arguments.
        :return: List of embeddings for the given corpus.
        """
        if isinstance(corpus, dict):
            sentences = [
                (
                    (corpus["title"][i] + self.sep + corpus["text"][i]).strip()
                    if "title" in corpus
                    else corpus["text"][i].strip()
                )
                for i in range(len(corpus["text"]))
            ]
        else:
            sentences = [
                (
                    (doc["title"] + self.sep + doc["text"]).strip()
                    if "title" in doc
                    else doc["text"].strip()
                )
                for doc in corpus
            ]
        return self.encode(sentences, batch_size=batch_size, **kwargs)
