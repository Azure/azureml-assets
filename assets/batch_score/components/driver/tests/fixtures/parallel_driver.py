# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock parallel driver."""

import asyncio

import pytest

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.parallel.parallel_driver import Parallel


@pytest.fixture()
def make_parallel_driver(make_conductor, make_input_transformer):
    """Mock parallel driver."""
    default_loop = asyncio.new_event_loop()

    def make(
        loop=None,
        conductor=None,
        input_to_request_transformer=None,
        input_to_log_transformer=None,
        input_to_output_transformer=None,
    ):
        loop = loop or default_loop
        return Parallel(
            configuration=Configuration(),
            loop=loop,
            conductor=conductor or make_conductor(loop=loop),
            input_to_request_transformer=input_to_request_transformer or make_input_transformer(),
            input_to_log_transformer=input_to_log_transformer or make_input_transformer(),
            input_to_output_transformer=input_to_output_transformer or make_input_transformer(),
        )

    return make
