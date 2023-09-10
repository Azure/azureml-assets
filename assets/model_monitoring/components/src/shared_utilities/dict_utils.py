# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains additional utilities that are applicable to python dictionaries."""


def merge_dicts(source_dict, new_dict):
    """Merge two dictionaries."""
    result = source_dict

    if type(source_dict) is not dict or type(new_dict) is not dict:
        print(source_dict, new_dict)
        raise Exception(
            f"Error: Attempting to merge non-dictionary objects {source_dict} and {new_dict}."
        )

    for key in new_dict:
        if key in source_dict:
            result[key] = _merge_dicts(source_dict[key], new_dict[key], key)
        else:
            result[key] = new_dict[key]

    return result


def _merge_dicts(source_dict, new_dict, key_scope):
    result = source_dict

    if type(source_dict) is not dict or type(new_dict) is not dict:
        print(source_dict, new_dict)
        raise Exception(
            f"Error: A duplicate key is found for entries {{{key_scope}: {new_dict}}} and"
            + f" {{{key_scope}: {source_dict}}}"
        )

    for key in new_dict:
        if key in source_dict:
            result[key] = _merge_dicts(
                source_dict[key], new_dict[key], f"{key_scope}.{key}"
            )
        else:
            result[key] = new_dict[key]

    return result
