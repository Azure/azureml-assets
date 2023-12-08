# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains a mock PRS mini batch context class."""

class TestMiniBatchContext:
    """Mock PRS mini batch context class."""
    __test__ = False

    def __init__(self, minibatch_index):
        """Init function."""
        self.minibatch_index = minibatch_index
