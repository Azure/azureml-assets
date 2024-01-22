# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the unit tests for MIR and batch pool header handler factory."""

import pytest

from src.batch_score.common.configuration.configuration_parser import ConfigurationParser
from src.batch_score.common.configuration.configuration import Configuration
from src.batch_score.header_handlers.mir_and_batch_pool_header_handler_factory import (
    MirAndBatchPoolHeaderHandlerFactory
)
from src.batch_score.header_handlers.open_ai import (
    ChatCompletionHeaderHandler,
    CompletionHeaderHandler,
    OpenAIHeaderHandler,
    SaharaHeaderHandler,
    VestaHeaderHandler,
)
from src.batch_score.header_handlers.open_ai.vesta_chat_completion_header_handler import (
    VestaChatCompletionHeaderHandler
)


@pytest.mark.parametrize('method_to_return_true, expected_header_handler_type', [
    ('is_sahara', SaharaHeaderHandler),
    ('is_vesta', VestaHeaderHandler),
    ('is_vesta_chat_completion', VestaChatCompletionHeaderHandler),
    ('is_completion', CompletionHeaderHandler),
    ('is_embeddings', CompletionHeaderHandler),
    ('is_chat_completion', ChatCompletionHeaderHandler),
    ('', OpenAIHeaderHandler)
])
def test_get_header_handler(mocker, make_metadata, method_to_return_true, expected_header_handler_type):
    # Arrange
    mocker.patch.object(Configuration, 'is_sahara', return_value='is_sahara' == method_to_return_true)
    mocker.patch.object(Configuration, 'is_vesta', return_value='is_vesta' == method_to_return_true)
    mocker.patch.object(Configuration, 'is_vesta_chat_completion',
                        return_value='is_vesta_chat_completion' == method_to_return_true)
    mocker.patch.object(Configuration, 'is_completion', return_value='is_completion' == method_to_return_true)
    mocker.patch.object(Configuration, 'is_embeddings', return_value='is_embeddings' == method_to_return_true)
    mocker.patch.object(Configuration, 'is_chat_completion',
                        return_value='is_chat_completion' == method_to_return_true)
    configuration = ConfigurationParser().parse_configuration([])

    # Act
    header_handler = MirAndBatchPoolHeaderHandlerFactory().get_header_handler(
        configuration=configuration,
        metadata=make_metadata,
        routing_client=None,
        token_provider=None
    )

    # Assert
    assert isinstance(header_handler, expected_header_handler_type)
