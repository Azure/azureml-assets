# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Mini batch context."""


class MiniBatchContext(object):
    """Mini batch context."""

    def __init__(self, raw_mini_batch_context, target_result_len) -> None:
        """Initialize MiniBatchContext."""
        self.__target_result_len = target_result_len
        self.raw_mini_batch_context = raw_mini_batch_context
        self.exception = None

    @property
    def mini_batch_id(self):
        """Identifier of the mini batch."""
        return self.raw_mini_batch_context.minibatch_index

    @property
    def target_result_len(self):
        """Target result length of the mini batch."""
        return self.__target_result_len
