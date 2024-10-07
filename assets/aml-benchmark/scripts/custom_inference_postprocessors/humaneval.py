# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------
"""Custom inference postprocessor for HumanEval."""
import re
import json
import codecs
import argparse
import pandas as pd
import sys
from io import StringIO
from jinja2 import Environment
from datasets import load_dataset
from typing import Any, Dict, List, Optional, Union, Tuple

JINJA_ENV = Environment(keep_trailing_newline=True)
REGEX_EXPR = """((?:.*?def.*?FUNCNAME.*?))(?=(?:
\\S|$))"""
CODE_GENERATION_DEBUG = False


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
    parser.add_argument(
        "--additional_parameters",
        type=str,
        default='null',
        help="Additional parameter values set in other fields of the component in a pipeline."
    )
    argss = parser.parse_args()
    return argss


def _read_input_file(file_path: str) -> List[Dict[str, Any]]:
    """Read files that have content in the json format and return a list of dictionaries.

    :param file_paths: Path to .jsonl file.
    :return: List of dictionaries.
    """
    data_dicts = []
    with open(file_path, "r", encoding="utf8") as file:
        for line in file:
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


def _extract_text_from_markdown_tag(input_string: str, tag_type='python') -> str:
    """
    Extract text between markdown code tags.

    If the tag pattern search returns no matches, the input string is returned.
    """
    pattern = f"```{tag_type}(.*?)```"
    m = re.search(pattern, input_string, flags=re.DOTALL)
    if not m:
        pattern_partial = f"```{tag_type}(.*)"
        m = re.search(pattern_partial, input_string, flags=re.DOTALL)
    return m.group(1) if m else input_string


def _run(
    prediction_dataset: str,
    output_path: str,
    ground_truth_dataset: str = None,
    additional_args: Optional[dict] = None,
) -> None:
    """Entry function to read, run and write the processed the data."""
    if prediction_dataset:
        pred_data = _read_input_file(prediction_dataset)
        if ground_truth_dataset:
            task_id = _read_input_file(ground_truth_dataset)
            pred_with_task_id, label_key, prediction_key = merge_id(task_id, pred_data, additional_args)

    # Post processing the prediction and ground truth columns
    ground_truths, predictions = run_humaneval_postprocessor(pred_with_task_id,
                                                             label_key,
                                                             prediction_key,
                                                             REGEX_EXPR)
    _write_to_jsonl_file(predictions, output_path, ground_truths)


def merge_id(
    label_data: List[Dict[str, Any]],
    pred_data: List[Dict[str, Any]],
    additional_args=None,
) -> Tuple[List[Dict[Any, Any]], str, str]:
    """
    Merge the task_id with the prediction data.

    :param label_data: Label data loaded from _read_input_file function.
    :type: List[Dict[str, Any]]
    :param pred_data: Prediction data loaded from _read_input_file function.
    :type: List[Dict[str, Any]]
    :param additional_args: Additional arguments passed to the function.
    :type: Optional[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    additional_args = {} if additional_args is None else additional_args

    label_key = additional_args.get("ground_truth_column_name", None)
    prediction_key = additional_args.get("prediction_column_name", None)

    if label_key is None:
        label_key = next(iter(label_data[0]))
    if prediction_key is None:
        prediction_key = next(iter(pred_data[0]))

    expected_op = [{**x, **y} for x, y in zip(label_data, pred_data)]
    return expected_op, label_key, prediction_key


def run_humaneval_postprocessor(
    data: List[Dict[str, Any]],
    label_key: str,
    prediction_key: str,
    regex_exp: str,
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom post processor function to extract the expected code.

    :param data: Model prediction data
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
                            left_on=label_key,
                            right_on='task_id')

    # Rename the prediction column
    pred_df_full.rename(columns={prediction_key: "original_prediction"},
                        inplace=True)

    # Convert the dataframe to a dictionary of lists of each row
    pred_dict_full = pred_df_full.to_dict('records')

    # Post processing the prediction and ground truth columns
    for row in pred_dict_full:
        gt = "\n" + row["test"] + "\n" + "check(" + row["entry_point"] + ")"
        tag_type = "python" if "```python" in row["original_prediction"] else ""

        # Extract the prediction data from the markdown tags
        pred_combined_prompt = _extract_text_from_markdown_tag(row["original_prediction"], tag_type)

        # Get the index of the first function definition and return keyword
        def_index = pred_combined_prompt.find("def ")
        func_name_index = pred_combined_prompt.find(str("def " + row["entry_point"]))
        return_keyword_index = pred_combined_prompt.find("return")

        # If function definition is not present or present after the initial function body prediction
        if def_index == -1 or return_keyword_index < def_index or func_name_index == -1:
            # If spaces were stripped from endpoint responses, add those back.
            if len(pred_combined_prompt) > 0 and pred_combined_prompt[0].isspace():
                prefix = ""
            else:
                prefix = "    "
            pred_combined_prompt = row["prompt"] + "\n" + prefix + pred_combined_prompt
        else:
            # If function name is present in the prediction, then remove from prompt
            prompt_header = row["prompt"].split(str("def " + row["entry_point"]))[0]
            # Removing spaces from the beginning of the prediction
            if len(pred_combined_prompt) > 0 and pred_combined_prompt[0].isspace():
                pred_combined_prompt = pred_combined_prompt.lstrip()
            pred_combined_prompt = prompt_header + "\n" + pred_combined_prompt

        # Applying regex on the prediction column
        if regex_exp:
            regex_exp_func = regex_exp.replace("FUNCNAME", row["entry_point"])
            pred = apply_regex_expr(pred_combined_prompt, regex_exp_func)
        else:
            pred = pred_combined_prompt
        if CODE_GENERATION_DEBUG is True:
            detailed_op = generate_output(pred,
                                          gt,
                                          row["task_id"],
                                          pred_combined_prompt,
                                          row["original_prediction"])
            gt_list.append(detailed_op)
        else:
            gt_list.append({"ground_truth": gt})
        pred_list.append({"prediction": pred})
    return gt_list, pred_list


def generate_output(
        pred: str,
        test_cases: str,
        index: str,
        pred_combined_prompt: str,
        pred_orig: str) -> Dict[str, Any]:
    """To debug the python codes."""
    op_details = {
        "index": index,
        "pred_orig": pred_orig,
        "pred_combined_prompt": pred_combined_prompt,
        "pred_with_regex": pred,
        "ground_truth": test_cases,
        "full_code": str(pred + test_cases),
        }
    if CODE_GENERATION_DEBUG is True:
        output, error, error_type = run_code(pred+test_cases)
        op_details["error"] = error
        op_details["error_type"] = error_type
        op_details["output"] = output
    return op_details


def run_code(code: str):
    """To run the test cases."""
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = StringIO()
    sys.stderr = StringIO()

    output = None
    error = None
    error_type = None

    try:
        exec(code, {})
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
        regex_exp: str
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
    _run(
        argss.prediction_dataset, argss.output_dataset,
        argss.ground_truth_dataset, json.loads(argss.additional_parameters)
    )
