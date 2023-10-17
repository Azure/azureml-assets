# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Custom preprocessor script for dataset:TruthfulQA, source:HF, config:generation."""

from typing import Any, Dict, List, Union
import argparse
import json
import pandas as pd
from random import shuffle


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


def format_string(sentence: str) -> str:
    """Apply formatting at the end of the text and adds fullstop if it does not ned with it."""
    if not sentence.endswith("."):
        sentence = sentence + "."
    return sentence


def run_processor(data: List[Dict[str, Any]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """Run the custom processor function."""
    # generic preprocessor
    input_keys = ["question", "best_answer", "incorrect_answers"]
    label_key = "correct_answers"

    # create following dictionary for labels in format {'1': 'A', '2': 'B', ... '26': 'Z'}
    encoder_config = {str(i): chr(i+ord('A')-1) for i in range(1, 27)}
    ret_data = []
    for example in data:
        out_dict = {}
        choices = []
        for key in input_keys:
            if key == "best_answer":
                out_dict['best_answer'] = format(example.get(key, '').strip())
                choices.append(format(example.get(key, '').strip()))
                shuffle(choices)
            elif key == 'incorrect_answers':
                choices += [format(i.strip()) for i in example.get("incorrect_answers")]
                shuffle(choices)
            else:
                out_dict[key] = format(example.get(key, '').strip())
        out_dict['choices'] = choices
        out_dict['best_answer_index'] = str(choices.index(out_dict['best_answer'])+1)
        out_dict['correct_answers'] = [format(i.strip()) for i in example.get(label_key)]
        out_dict['labels'] = [chr(i+ord('A')) for i in range(len(choices))]
        out_dict['best_answer_label'] = encoder_config.get(str(out_dict.get('best_answer_index')))
        ret_data.append(out_dict)
    return ret_data


if __name__ == '__main__':
    argss = _parse_args()
    _run(argss.input_path, argss.output_path)
