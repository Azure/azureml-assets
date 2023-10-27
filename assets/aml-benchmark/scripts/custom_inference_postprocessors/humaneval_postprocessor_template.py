# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Custom inference postprocessor for HumanEval."""
import re
import json
import codecs
import argparse
import logging as logger
import pandas as pd
import sys

from io import StringIO
from jinja2 import Environment
from datasets import load_dataset
from typing import Any, Dict, List, Union

# flake8: noqa
JINJA_ENV = Environment(keep_trailing_newline=True)
REGEX_EXPR = """((?:.*?def(?=.*?(decode|find_zero|make_palindrome)).*?def.*?|.*?def.*?))(?=(?:
\S|$))"""  
failed_runs = []


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
    ground_truth_dataset: str = None,
    regex_expr: str = None
) -> None:
    """Entry function to read, run and write the processed the data."""
    if prediction_dataset:
        pred_data = _read_jsonl_file(prediction_dataset)
        if ground_truth_dataset:
            task_id = _read_jsonl_file(ground_truth_dataset)
            pred_with_task_id = merge_id(task_id, pred_data)

    regex_expr = REGEX_EXPR

    # Post processing the prediction and ground truth columns
    ground_truths, predictions = run_humaneval_postprocessor(pred_with_task_id, regex_expr)

    _write_to_jsonl_file(predictions, output_path, ground_truths)


def merge_id(
    label_data: List[Dict[str, Any]],
    pred_data: List[Dict[str, Any]]
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function to extract the ground truth.

    :param data: Data loaded from _read_jsonl_file function.
    :type: List[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    # Get the original data from Hugging Face
    expected_op = [{**x, **y} for x, y in zip(label_data, pred_data)]
    return expected_op


def run_humaneval_postprocessor(
    data: List[Dict[str, Any]],
    regex_exp: str = None
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function to extract the ground truth.

    :param data: Data loaded from _read_jsonl_file function.
    :type: List[Dict[str, Any]]
    :param regex_exp: Regex expression to extract the prediction.
    :type: str
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    gt_list = []
    pred_list = []

    # Get the original data from Hugging Face
    original_dataset = load_dataset("openai_humaneval", split="test").to_pandas()
    pred_df = pd.DataFrame(data)

    # Merge the prediction dataset with the original dataset
    pred_df_full = pd.merge(pred_df,
                            original_dataset,
                            how='left',
                            left_on='ground_truth',
                            right_on='task_id')

    # Rename the prediction column
    pred_df_full.rename(columns={"prediction": "original_prediction"},
                        inplace=True)

    # Convert the dataframe to a dictionary of lists of each row
    pred_dict_full = pred_df_full.to_dict('records')

    # Post processing the prediction and ground truth columns
    for idx, row in enumerate(pred_dict_full):
        gt = "\n" + pred_dict_full[idx]["test"] + "\n" + "check(" + pred_dict_full[idx]["entry_point"] + ")"

        if str("def " + pred_dict_full[idx]["entry_point"] + "(") in pred_dict_full[idx]["original_prediction"]:
            # If the model regenerates the prompt/ function name
            pred_combined_prompt = pred_dict_full[idx]["original_prediction"]
        else:
            pred_combined_prompt = pred_dict_full[idx]["prompt"]+"\n"+pred_dict_full[idx]["original_prediction"]

        # Applying regex on the prediction column
        if regex_exp:
            pred = apply_regex_expr(pred_combined_prompt, regex_exp)
        else:
            pred = pred_combined_prompt
        failed, _ = code_run(pred,
                             gt,
                             pred_dict_full[idx]["task_id"],
                             pred_combined_prompt,
                             pred_dict_full[idx]["original_prediction"])
        gt_list.append(failed)
        # gt_list.append({"ground_truth": gt})
        pred_list.append({"prediction": pred})
    return gt_list, pred_list


def code_run(pred, test_cases, index, pred_combined_prompt, pred_orig):
    failed_runs = []
    output, error, error_type = run_code(pred+test_cases)
    failed = {
        "index": index,
        "pred_orig": pred_orig,
        "pred_combined_prompt": pred_combined_prompt,
        "pred_final_with_regex": pred,
        "ground_truth": test_cases,

        "full_code": str(pred + test_cases),
        "error": error,
        "error_type": error_type,
        "output": output,
        }
    failed_runs.append(failed)
    return failed, failed_runs


def run_code(code):
    """Function to run code."""

    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    output = None
    error = None
    error_type = None

    try:
        exec(code)
        output = sys.stdout.getvalue()
        output = "No error, executed successfully"
    except Exception as e:
        error = sys.stderr.getvalue()
        if not error:
            error = str(e)
        error_type = type(e).__name__

    # Restore stdout and stderr
    sys.stdout = original_stdout
    sys.stderr = original_stderr

    return output, error, error_type


def apply_regex_expr(
        text: str,
        regex_exp: str = None
        ) -> str:
    """Apply regex on the given text."""
    if regex_exp:
        regex_exp = codecs.decode(regex_exp, "unicode_escape")
        matches = re.search(regex_exp, text, flags=re.DOTALL)
        if matches is None or len(matches.groups()) == 0:
            return text
        return matches.group(1)
    return text


if __name__ == "__main__":
    argss = _parse_args()
    _run(argss.prediction_dataset, argss.output_dataset, argss.ground_truth_dataset)
