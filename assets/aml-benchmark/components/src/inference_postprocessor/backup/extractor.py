import codecs
import json
import re
import logging
from typing import List, Callable, Dict
import numpy as np
from collections import OrderedDict


logger = logging.getLogger(__name__)
OUTPUT_FILENAME = "extracted_data.jsonl"

def apply_separator(completion, separator, keep_separator):
    append_separator = keep_separator and separator in completion
    return completion.split(separator)[0] + (separator if append_separator else "")

def _convert_to_unicode(text):
    """Convert from a raw string to a unicode string.

    Example:
        >>> "\nExample".startswith(r"\n") # False
        >>> "\nExample".startswith(codecs.decode(r"\n", "unicode_escape")) # True
    """
    return codecs.decode(text, "unicode_escape")

def remove_prefix(completion, prefix):
    if completion.startswith(prefix):
        completion = completion[len(prefix):]
    else:
        prefix = _convert_to_unicode(prefix)
        completion = _convert_to_unicode(completion)
        if completion.startswith(prefix):
            completion = completion[len(prefix):]
    return completion

def remove_prefixes(completion, prefixes):
    prefixes = prefixes.split(",")
    for prefix in prefixes:
        completion = remove_prefix(completion, prefix)
    return completion

def apply_find_first(completion: str, candidates: List[str]) -> str:
    """Finds first occurence of any candidate in completion."""
    min_index = len(completion)
    first_candidate = ""
    for candidate in candidates:
        index = completion.find(candidate)
        if index != -1 and index < min_index:
            min_index = index
            first_candidate = candidate
    return first_candidate

def apply_extract_number(completion: str, strategy: str, default: str = "0") -> str:
    """Extracts first or last number from completion."""
    number_pattern = re.compile(r"(\-?[0-9\.\,\s]+)")
    match = number_pattern.findall(completion)
    if match:
        if strategy == "last":
            match = match[::-1]
        for m in match:
            if not re.search(r"\d", m):
                # we matched with a comma or full-stop, skip this
                continue
            else:
                m = m.strip()
                m = m.rstrip(".")
                # we only accept space and comma as separators of 3 digits in a number
                m = m.replace(" ", ",")
                m = m.strip(",")
                if "," in m:
                    parts = m.split(',')
                    if all(len(part) == 3 for part in parts[1:]):
                        m = ''.join(parts)
                    else:
                        m = parts[-1] if strategy == "last" else parts[0]
                try:
                    # Test that the matched string is a number
                    val = np.fromstring(m, sep=" ")
                    return m
                except SyntaxError:
                    # we matched with something that is not a number
                    pass
    return default

def apply_replacements(completion, replacements):
    replacements_map = _parse_string_into_dict(replacements)

    for previous, new in replacements_map.items():
        previous = _convert_to_unicode(previous)
        new = _convert_to_unicode(new)
        completion = completion.replace(previous, new)
    return completion

def extract_regex(completion, regex):
    completion_search = re.search(regex, completion, flags=re.DOTALL)

    if completion_search is None or len(completion_search.groups()) == 0:
        return completion
    completion = completion_search.group(1)

    return completion

def create_label_map(label_map: str):
    return _parse_string_into_dict(label_map)

def _parse_string_into_dict(string: str) -> OrderedDict:
    """Parse a string into a dictionary."""
    if not string:
        return {}
    
    replacements = OrderedDict()

    for replacement in string.split(","):
        key_value = replacement.split(":")
        if len(key_value) != 2:
            logger.error(f"Invalid replacement: {replacement}")
            continue
        previous, new = key_value
        replacements[previous] = new
    
    return replacements

def apply_label_map(completion, label_map):
    """Find first label in completion and apply label map to it."""
    candidates = [k for k in label_map if k in completion]
    if len(candidates) == 0:
        return str(len(label_map))
    else:
        first_candidate = min(candidates, key=lambda x: completion.find(x))
        return str(label_map.get(first_candidate))

def unpack_with_adjustment(line: str, adjustment: Callable[[Dict], Dict]) -> Dict:
    data = adjustment(json.loads(line))

    # flatten metadata
    if "metadata" in data:
        for k, v in data["metadata"].items():
            # Avoid accidental override of key in data
            key = f"{k}_metadata" if k in data else k
            data[key] = v
        del data["metadata"]
    
    if '_batch_request_metadata' in data:
        for k, v in data["_batch_request_metadata"].items():
            # Avoid accidental override of key in data
            key = f"{k}_metadata" if k in data else k
            data[key] = v
        del data["_batch_request_metadata"]
    
    return data

def batch_score_response_format_adjustment(data, completion_key="samples"):
    """
    Because the response format is different between the scoring components,
    we need to adjust the schema for batch_score to be in line with other Babel components.
    """
    try:
        new_data = {
            "prompt": data["request"]["prompt"],
            completion_key: [sample["text"] for sample in data["response"]["choices"]],
        }
        if "request_metadata" in data:
            new_data["metadata"] = data["request_metadata"]
            if "completion" in new_data["metadata"]:
                new_data["completion"] = new_data["metadata"]["completion"]
    except Exception:
        parsed_response = json.loads(data["response"])
        if "error" in parsed_response:
            logger.error(f"Error returned by the endpoint:\n{parsed_response['error']}")
        else:
            logger.exception("Something went wrong while converting schema.")
        new_data = data
    return new_data
