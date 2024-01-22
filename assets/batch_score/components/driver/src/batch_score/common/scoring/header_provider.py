# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the definition for abstract header provider."""

from abc import abstractmethod


class HeaderProvider:
    @abstractmethod
    def get_headers(self) -> dict:
        pass