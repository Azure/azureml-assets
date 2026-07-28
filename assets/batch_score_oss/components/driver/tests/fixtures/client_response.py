# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains fixtures to mock a client response."""


class FakeResponse:
    """Mock response."""

    def __init__(
            self,
            status,
            json,
            *,
            headers=None,
            text=None,
            error=None,
            reason=None):
        """Initialize FakeResponse."""
        self.status = status
        self.headers = headers or {}
        self.reason = reason or ''
        self._json = json
        self._text = text or ""
        self._error = error

    async def json(self):
        """Get json."""
        return self._json

    async def text(self):
        """Get text."""
        return self._text

    def raise_for_status(self):
        """Raise error."""
        if self._error is not None:
            raise self._error
