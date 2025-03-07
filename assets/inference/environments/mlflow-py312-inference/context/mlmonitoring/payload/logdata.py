"""For logdata."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


def get_type_fullname(o):
    """For get type fullname."""
    c = o.__class__
    module = c.__module__
    if module == 'builtins':
        return "python." + c.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + c.__qualname__


class LogData(dict):
    """For LogData."""

    def __init__(self, data):
        """For init."""
        dict.__init__(self)
        self._data = data

    def type(self):
        """For type."""
        return get_type_fullname(self._data)

    def to_json(self):
        """For to json."""
        pass

    def to_bytes(self):
        """For to bytes."""
        pass


class PandasFrameData(LogData):
    """For PandasFrameData."""

    def to_json(self):
        """For to json."""
        return self._data.to_json(orient="records")

    def to_bytes(self):
        """For to bytes."""
        return bytes(self._data.to_string(), "utf-8")
