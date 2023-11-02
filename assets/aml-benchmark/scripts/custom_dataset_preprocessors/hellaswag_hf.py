# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Custom preprocessor script for dataset:Hellaswag, source:HF."""

from typing import Any, Dict, List, Union
import argparse
import json
import pandas as pd
import re


def _parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    argss = parser.parse_args()
    return argss


def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read `.jsonl` file and return a list of dictionaries."""
    if not file_path.endswith(".jsonl"):
        mssg = f"Input file '{file_path}' is not a .jsonl file."
        raise ValueError(mssg)
    data_dicts = []
    with open(file_path, 'r', encoding="utf8") as file:
        for i, line in enumerate(file):
            data_dicts.append(json.loads(line))
    return data_dicts


def _write_to_jsonl_file(data, file_path: str) -> None:
    """Write the processed output to jsonl file."""
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, lines=True, orient='records')
        return
    if isinstance(data, List):
        with open(file_path, 'w') as writer:
            for example in data:
                writer.write(json.dumps(example) + "\n")
    return


def _run(input_path: str, output_path: str) -> None:
    """Entry function to read, run and write the processed the data."""
    data = _read_jsonl_file(input_path)
    processed_data = run_processor(data)
    _write_to_jsonl_file(processed_data, output_path)


def preprocess(text):
    """Clean the data."""
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text


def run_processor(data: List[Dict[str, Any]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Run the custom processor function."""
    ret_data = []
    for doc in data:
        ctx = doc["ctx_a"] + " " + doc["ctx_b"].capitalize()
        out_doc = {
            "query": preprocess(doc["activity_label"] + ": " + ctx),
            "choices": [preprocess(ending) for ending in doc["endings"]],
            "gold": str(doc["label"]),
        }
        ret_data.append(out_doc)
    return ret_data


if __name__ == '__main__':
    argss = _parse_args()
    _run(argss.input_path, argss.output_path)
