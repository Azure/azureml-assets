# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

from typing import Any, Dict, List, Union
import argparse
import json
import logging as logger
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    argss = parser.parse_args()
    return argss


def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Read `.jsonl` file and return a list of dictionaries.
    :param file_paths: Path to .jsonl file.
    :return: List of dictionaries.
    """
    if not file_path.endswith(".jsonl"):
        mssg = f"Input file '{file_path}' is not a .jsonl file."
        logger.ERROR(mssg)
        raise ValueError(mssg)
    data_dicts = []
    with open(file_path, 'r') as file:
        for i, line in enumerate(file):
            data_dicts.append(json.loads(line))
    return data_dicts


def _write_to_jsonl_file(data, file_path: str) -> None:
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, lines=True, orient='records')
        return
    if isinstance(data, List):
        with open(file_path, 'w') as writer:
            for example in data:
                writer.write(json.dumps(example) + "\n")
    return


def _run(input_path: str, output_path: str) -> None:
    data = _read_jsonl_file(input_path)
    processed_data = run_processor(data)
    _write_to_jsonl_file(processed_data, output_path)


def run_processor(data: List[Dict[str, Any]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    This is the function where user needs to write their preprocessor logic.
    :param input_path: path to the jsonl input file
    :param output_path: path to the jsonl output file
    """
    from random import shuffle
    # generic preprocessor
    input_keys = list(map(lambda x: x.strip(), "question,best_answer,incorrect_answers".split(',')))
    label_key = "correct_answers"

    def format(sentence: str) -> str:
        if sentence[-1] != ".":
            sentence = sentence + "."
        return sentence

    encoder_config = {
        "1": "A", "2": "B", "3": "C", "4": "D", "5": "E", "6": "F",
        "7": "G", "8": "H", "9": "I", "10": "J", "11": "K", "12": "L",
        "13": "M", "14": "N", "15": "O", "16": "P"
    }
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
        if encoder_config:
            out_dict['best_answer_label'] = encoder_config.get(str(out_dict.get('best_answer_index')))
        ret_data.append(out_dict)
    return ret_data


if __name__ == '__main__':
    argss = _parse_args()
    _run(argss.input_path, argss.output_path)
