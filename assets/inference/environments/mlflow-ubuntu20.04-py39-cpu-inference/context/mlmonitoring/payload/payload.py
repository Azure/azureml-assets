"""For payload."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import uuid

from .logdata import LogData
from ..logger import logger
from ..common import __version__
from ..config import get_config


class Payload:
    """For payload class."""

    def __init__(self, designation, data, model_version, timestamp, correlation_id, headers):
        """For init."""
        self._id = str(uuid.uuid4())
        self._designation = designation
        self._model_version = model_version
        self._agent = "azureml-ai-monitoring/" + __version__
        self._contenttype = ""

        if not isinstance(data, LogData):
            raise TypeError("argument data (%s) must be one of LogData types." % type(data).__name__)

        self._data = data
        self._correlation_id = correlation_id
        if timestamp is not None:
            self._time = timestamp
        else:
            self._time = int(time.time())
        self._headers = headers

    def time(self):
        """For time."""
        return self._time

    def id(self):
        """For id."""
        return self._id

    def designation(self):
        """For designation."""
        return self._designation

    def model_version(self):
        """For model version."""
        return self._model_version

    def agent(self):
        """For agent."""
        return self._agent

    def correlation_id(self):
        """For correlation id."""
        return self._correlation_id

    def headers(self):
        """For headers."""
        return self._headers

    def type(self):
        """For type."""
        return self._data.type()

    def contenttype(self):
        """For contenttype."""
        return self._contenttype

    def content(self):
        """For content."""
        pass


class JsonPayload(Payload):
    """For JsonPayload."""

    def __init__(self, designation, data, model_version=None, timestamp=None, correlation_id=None, headers=None):
        """For init."""
        if headers is None:
            headers = {}
        super().__init__(designation, data, model_version, timestamp, correlation_id, headers)
        self._contenttype = "application/json"

    def content(self):
        """For content."""
        return self._data.to_json()


class CompactPayload(Payload):
    """For CompactPayload."""

    def __init__(self, designation, data, model_version=None, timestamp=None, correlation_id=None, headers=None):
        """For init."""
        if headers is None:
            headers = {}
        super().__init__(designation, data, model_version, timestamp, correlation_id, headers)
        self._contenttype = "application/octet-stream"

    def content(self):
        """For content."""
        return self._data.to_bytes()


def build_payload(designation, data, model_version=None, context=None):
    """For build payload."""
    logger.debug("building payload for collection %s, data type: %s", designation, type(data).__name__)
    headers = {}
    timestamp = None
    correlation_id = None
    if context is not None:
        correlation_id = context.get_id()
        timestamp = context.get_timestamp()
        headers = context.get_headers()

    config = get_config()
    if config.compact_format():
        return CompactPayload(
            designation,
            data,
            model_version=model_version,
            correlation_id=correlation_id,
            timestamp=timestamp,
            headers=headers)

    return JsonPayload(
        designation,
        data,
        model_version=model_version,
        correlation_id=correlation_id,
        timestamp=timestamp,
        headers=headers)
