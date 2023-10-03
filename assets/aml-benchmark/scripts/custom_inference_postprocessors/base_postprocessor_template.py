# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Base template for custom inference postprocessor script."""

from typing import Any, Dict, List, Union
import argparse
import json
import logging as logger
import pandas as pd


def _parse_args():
    """Parse the arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prediction_dataset",
        type=str,
        required=True,
        help="Path to load the prediction dataset."
    )
    parser.add_argument(
        "--ground_truth_dataset",
        type=str,
        help="Path to load the actual dataset."
    )
    parser.add_argument(
        "--output_dataset",
        type=str,
        help="Path to the jsonl output file to write the processed data."
    )
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
    prediction: Union[pd.DataFrame, List[Dict[str, Any]]],
    file_path: str,
    ground_truth: Union[pd.DataFrame, List[Dict[str, Any]], None] = None,
) -> None:
    """Write the processed output to jsonl file.

    :param data: Data to write to the output file provided in `file_path`.
    :param file_path: Path of the output file to dump the data.
    :return: None
    """
    if isinstance(prediction, pd.DataFrame):
        if ground_truth and isinstance(ground_truth, pd.DataFrame):
            pd.concat([ground_truth, prediction], axis=1).to_json(file_path, lines=True, orient="records")
        else:
            prediction.to_json(file_path, lines=True, orient="records")
        return
    if isinstance(prediction, List):
        if ground_truth and isinstance(ground_truth, List):
            with open(file_path, "w") as writer:
                for i in range(0, len(prediction)):
                    example = prediction[i]
                    example.update(ground_truth[i])
                    writer.write(json.dumps(example) + "\n")
        else:
            with open(file_path, "w") as writer:
                for example in prediction:
                    writer.write(json.dumps(example) + "\n")
    return


def _run(
    prediction_dataset: str,
    output_path: str,
    ground_truth_dataset: str = None
) -> None:
    """Entry function to read, run and write the processed the data."""
    pred_data = _read_jsonl_file(prediction_dataset)
    predictions = run_prediction_extractor(pred_data)
    if ground_truth_dataset:
        actual_data = _read_jsonl_file(ground_truth_dataset)
        ground_truths = run_ground_truth_extractor(actual_data)
    _write_to_jsonl_file(predictions, ground_truths, output_path)


def run_ground_truth_extractor(
    data: List[Dict[str, Any]]
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function to extract the ground truth.

    :param data: Data loaded from _read_jsonl_file function.
    :type: List[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    pass


def run_prediction_extractor(
    data: List[Dict[str, Any]]
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function to extract the ground truth.

    :param data: Data loaded from _read_jsonl_file function.
    :type: List[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    # write the preprocessing logic here
    pass


if __name__ == "__main__":
    argss = _parse_args()
    _run(argss.prediction_dataset, argss.ground_truth_dataset, argss.output_dataset)
