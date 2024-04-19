# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

def get_type_fullname(o):
    c = o.__class__
    module = c.__module__
    if module == 'builtins':
        return "python." + c.__qualname__  # avoid outputs like 'builtins.str'
    return module + '.' + c.__qualname__


class LogData(dict):
    def __init__(self, data):
        dict.__init__(self)
        self._data = data

    def type(self):
        return get_type_fullname(self._data)

    def to_json(self):
        pass

    def to_bytes(self):
        pass


class PandasFrameData(LogData):

    def to_json(self):
        return self._data.to_json(orient="records")

    def to_bytes(self):
        return bytes(self._data.to_string(), "utf-8")
