# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

from typing import Any, Dict, List, Union
import argparse
import json
import logging as logger

def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--output_path', type=str, required=True)
    args = parser.parse_args()
    return args

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

def _write_to_jsonl_file(data, file_path:str) -> None:
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
    _write_to_jsonl_file(data, output_path)

def run_processor(data:List[Dict[str, Any]]) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    This is the function where user needs to write their preprocessor logic.
    :param input_path: path to the jsonl input file
    :param output_path: path to the jsonl output file
    """
    # write your pre-processing logic
    


if __name__ == '__main__':
    args = _parse_args()
    _run(args.input_path, args.output_path)