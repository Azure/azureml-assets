import hashlib
from typing import Dict, Any

class SHA256Checksum:
    def __init__(self):
        self._hash = hashlib.sha256()

    def update(self, jsonl_line):
        line_as_str = self.__class__._prepare_jsonl_line(jsonl_line)
        self._hash.update(line_as_str)
    
    def digest(self):
        return self._hash.hexdigest()

    @staticmethod
    def _prepare_jsonl_line(jsonl_line: Dict[str,Any]):
        # Our components often need to read json lines into a dict
        # and then write it back into a file. This means we cannot rely on 
        # a specific order of the columns because they might have swapped order
        # (i.e., cannot just treat each line as a line). This is why we need to sort.
        if jsonl_line is None:
            result = ""
        else:
            sorted_by_key = [f"{k}:{v}" for k, v in sorted(jsonl_line.items())]
            result = "".join(sorted_by_key)

        # see https://stackoverflow.com/a/7585378
        return result.encode('utf-8')