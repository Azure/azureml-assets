# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Base template for custom inference postprocessor script."""

from typing import Any, Dict, List, Union, Optional
import argparse
import json
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
    parser.add_argument(
        "--additional_parameters",
        type=str,
        default='null',
        help="Additional parameter values set in other fields of the component in a pipeline."
    )
    argss = parser.parse_args()
    return argss


def _read_jsonl_file(file_path: str) -> List[Dict[str, Any]]:
    """Read `.jsonl` file and return a list of dictionaries.

    :param file_paths: Path to .jsonl file.
    :return: List of dictionaries.
    """
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
    additional_args: Optional[dict] = None,
) -> None:
    """Entry function to read, run and write the processed the data."""
    pred_data = _read_jsonl_file(prediction_dataset)
    predictions = run_prediction_extractor(pred_data, additional_args)
    if ground_truth_dataset:
        actual_data = _read_jsonl_file(ground_truth_dataset)
        ground_truths = run_ground_truth_extractor(actual_data, additional_args)
    _write_to_jsonl_file(predictions, output_path, ground_truths)


def run_ground_truth_extractor(
    data: List[Dict[str, Any]],
    additional_args: dict = None
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function to extract the ground truth.

    :param data: Data loaded from _read_jsonl_file function.
    :type: List[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    ret_data = []
    ground_truth_key = additional_args.get("ground_truth_column_name")
    for row in data:
        out_row = {}
        out_row[ground_truth_key] = row.get(ground_truth_key)
        ret_data.append(out_row)
    return ret_data


def run_prediction_extractor(
    data: List[Dict[str, Any]],
    additional_args: dict = None
) -> Union[pd.DataFrame, List[Dict[str, Any]]]:
    """
    Run the custom processor function to extract the ground truth.

    :param data: Data loaded from _read_jsonl_file function.
    :param additional_args: Additional parameters.
    :type: List[Dict[str, Any]]
    :return: pd.DataFrame or List[Dict[str, Any]]]
    """
    def remove_boxed(answer: str):
        left = "\\boxed{"
        try:
            assert answer[: len(left)] == left
            assert answer[-1] == "}"
            return answer[len(left):-1]
        except Exception:
            return None

    def last_boxed_only_string(string: str = None):
        idx = string.rfind("\\boxed")
        if idx < 0:
            idx = string.rfind("\\fbox")
            if idx < 0:
                return None
        i = idx
        right_brace_idx = None
        num_left_braces_open = 0
        while i < len(string):
            if string[i] == "{":
                num_left_braces_open += 1
            if string[i] == "}":
                num_left_braces_open -= 1
                if num_left_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1
        if right_brace_idx is None:
            retval = None
        else:
            retval = string[idx:right_brace_idx + 1]
        return retval

    def extract_final_answer(solution: str = None):
        if solution is None:
            return None
        # acc to paper, answer is in last box
        last_boxed = last_boxed_only_string(solution)
        if last_boxed is None:
            return None
        answer = remove_boxed(last_boxed)
        if answer is None:
            return None
        return answer

    def fix_fracs(string: str) -> str:
        substrs = string.split("\\frac")
        new_str = substrs[0]
        if len(substrs) > 1:
            substrs = substrs[1:]
            for substr in substrs:
                new_str += "\\frac"
                if substr[0] == "{":
                    new_str += substr
                else:
                    try:
                        assert len(substr) >= 2
                    except Exception:
                        return string
                    a = substr[0]
                    b = substr[1]
                    if b != "{":
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}{" + b + "}" + post_substr
                        else:
                            new_str += "{" + a + "}{" + b + "}"
                    else:
                        if len(substr) > 2:
                            post_substr = substr[2:]
                            new_str += "{" + a + "}" + b + post_substr
                        else:
                            new_str += "{" + a + "}" + b
        string = new_str
        return string

    def fix_a_slash_b(string: str) -> str:
        if len(string.split("/")) != 2:
            return string
        a_str = string.split("/")[0]
        b_str = string.split("/")[1]
        try:
            a = int(a_str)
            b = int(b_str)
            assert string == "{}/{}".format(a, b)
            new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
            return new_string
        except Exception:
            return string

    def remove_right_units(string: str) -> str:
        if "\\text{ " in string:
            splits = string.split("\\text{ ")
            # assert len(splits) == 2
            return splits[0]
        else:
            return string

    def fix_sqrt(string: str) -> str:
        if "\\sqrt" not in string:
            return string
        splits = string.split("\\sqrt")
        new_string = splits[0]
        for split in splits[1:]:
            if split[0] != "{":
                a = split[0]
                new_substr = "\\sqrt{" + a + "}" + split[1:]
            else:
                new_substr = "\\sqrt" + split
            new_string += new_substr
        return new_string

    def format(solution: str = None):
        if solution is None:
            return None
        # linebreaks
        solution = solution.replace("\n", "")
        # remove inverse spaces
        solution = solution.replace("\\!", "")
        # replace \\ with \
        solution = solution.replace("\\\\", "\\")
        # replace tfrac and dfrac with frac
        solution = solution.replace("tfrac", "frac")
        solution = solution.replace("dfrac", "frac")
        # remove \left and \right
        solution = solution.replace("\\left", "")
        solution = solution.replace("\\right", "")
        # Remove circ (degrees)
        solution = solution.replace("^{\\circ}", "")
        solution = solution.replace("^\\circ", "")
        # remove dollar signs
        solution = solution.replace("\\$", "")
        # remove units (on the right)
        solution = remove_right_units(solution)
        # remove percentage
        solution = solution.replace("\\%", "")
        solution = solution.replace(r"\%", "")
        # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively,
        # add "0" if "." is the start of the string
        solution = solution.replace(" .", " 0.")
        solution = solution.replace("{.", "{0.")
        # if empty, return empty string
        if len(solution) == 0:
            return solution
        if solution[0] == ".":
            solution = "0" + solution
        # to consider: get rid of e.g. "k = " or "q = " at beginning
        if len(solution.split("=")) == 2:
            if len(solution.split("=")[0]) <= 2:
                solution = solution.split("=")[1]
        # fix sqrt3 --> sqrt{3}
        solution = fix_sqrt(solution)
        # remove spaces
        # solution = solution.replace(" ", "")
        # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc.
        # Even works with \frac1{72} (but not \frac{72}1).
        # Also does a/b --> \\frac{a}{b}
        solution = fix_fracs(solution)
        # manually change 0.5 --> \frac{1}{2}
        if solution == "0.5":
            solution = "\\frac{1}{2}"
        # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
        solution = fix_a_slash_b(solution)
        return solution

    def _convert_to_unicode(text: str) -> str:
        import codecs
        return codecs.decode(text, "unicode_escape")

    ret_data = []
    pred_key = additional_args.get("prediction_column_name")
    for row in data:
        out_dict = {}
        shot_sep = _convert_to_unicode(additional_args.get("separator"))
        text = extract_final_answer(format(row.get(pred_key).split(shot_sep)[0]))
        out_dict[pred_key] = text if text is not None else ""
        ret_data.append(out_dict)
    return ret_data


if __name__ == "__main__":
    argss = _parse_args()
    _run(
        argss.prediction_dataset, argss.output_dataset,
        argss.ground_truth_dataset, json.loads(argss.additional_parameters)
    )
