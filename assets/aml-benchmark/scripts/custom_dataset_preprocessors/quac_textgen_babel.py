# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Quac text-gen preprocessor script."""

from typing import Any, Dict, List, Union
import argparse
import json
import logging as logger
import pandas as pd


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    argss = parser.parse_args()
    return argss


def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read `.jsonl` file and return a list of dictionaries.

    :param file_paths: Path to .jsonl file.
    :return: List of dictionaries.
    """
    if not file_path.endswith(".jsonl"):
        mssg = f"Input file '{file_path}' is not a .jsonl file."
        logger.ERROR(mssg)
        raise ValueError(mssg)
    data_dicts = []
    with open(file_path, "r", encoding="utf8") as file:
        for i, line in enumerate(file):
            data_dicts.append(json.loads(line))
    return data_dicts


def _write_to_jsonl_file(
    data: Union[pd.DataFrame, List[Dict[str, Any]]], file_path: str
) -> None:
    """Write the processed output to jsonl file.

    :param data: Data to write to the output file provided in `file_path`.
    :param file_path: Path of the output file to dump the data.
    :return: None
    """
    if isinstance(data, pd.DataFrame):
        data.to_json(file_path, lines=True, orient="records")
        return
    if isinstance(data, List):
        with open(file_path, "w") as writer:
            for example in data:
                writer.write(json.dumps(example) + "\n")
    return


def _run(input_path: str, output_path: str) -> None:
    """Entry function to read, run and write the processed the data."""
    data = _read_jsonl_file(input_path)
    processed_data = run_processor(data)
    _write_to_jsonl_file(processed_data, output_path)


def run_processor(
    data: List[Dict[str, Any]]
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function. The user needs to modify this function with their custom processing logic.

    :param data: Data loaded from _read_jsonl_file function.
    :type: List[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    ret_data = []
    data = data[0]
    for i, sample in enumerate(data["data"]):
        assert len(sample["paragraphs"]) == 1
        questions = [
            question
            for question in sample["paragraphs"][0]["qas"]
            if question["orig_answer"]["text"] != "CANNOTANSWER"
        ]
        if len(questions) == 0:
            continue
        possible_answers = [
            answer["text"]
            for answer in questions[-1]["answers"]
            if answer["text"] != "CANNOTANSWER"
        ]
        if questions[-1]["orig_answer"] != "CANNOTANSWER":
            possible_answers.append(questions[-1]["orig_answer"]["text"])
        if len(possible_answers) == 0:
            print(i)
            assert len(possible_answers) != 0
        ret_data.append(
            {
                "question": questions[-1]["question"],
                "answers": possible_answers,
                "context": sample["paragraphs"][0]["context"],
                "history": "\n".join(
                    [
                        question["question"] + "\n" + question["orig_answer"]["text"]
                        for question in questions[:-1]
                    ]
                ),
            }
        )
    return ret_data


if __name__ == "__main__":
    argss = _parse_args()
    _run(argss.input_path, argss.output_path)
