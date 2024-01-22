import asyncio

import pytest

from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.common.parallel.parallel_driver import Parallel


@pytest.fixture()
def make_parallel_driver(make_conductor, make_input_transformer):
    loop = asyncio.get_event_loop()

    def make(
        loop=loop,
        conductor=make_conductor(loop=loop),
        input_to_request_transformer=make_input_transformer(),
        input_to_log_transformer=make_input_transformer(),
        input_to_output_transformer=make_input_transformer(),
    ):
        return Parallel(
            configuration=Configuration(),
            loop=loop,
            conductor=conductor,
            input_to_request_transformer=input_to_request_transformer,
            input_to_log_transformer=input_to_log_transformer,
            input_to_output_transformer=input_to_output_transformer,
        )
    
    return make