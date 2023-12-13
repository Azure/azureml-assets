# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Math preprocessor script."""

from typing import Any, Dict, List, Union
import argparse
import json
import logging as logger
import pandas as pd


def _parse_args():
    """Parse the arguments."""
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
    with open(file_path, "r") as file:
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
    # write your pre-processing logic
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
            assert len(splits) == 2
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

    input_keys = ["problem"]
    label_key = "solution"

    try:
        df = pd.json_normalize(data)
        df = df[input_keys + [label_key]]
        for key in input_keys:
            df[key] = df[key].map(lambda x, y=format: y(x))
        df[label_key] = df[label_key].map(lambda x, y=format: y(x))
        df["answer"] = df[label_key].map(lambda x, y=extract_final_answer: y(x))
        return df
    except (RuntimeError, Exception):
        # fall back code
        ret_data = []
        for example in data:
            out_dict = {}
            for key in input_keys:
                out_dict[key] = format(example.get(key))
            out_dict[label_key] = format(example.get(label_key))
            out_dict["answer"] = extract_final_answer(out_dict.get(label_key))
            ret_data.append(out_dict)
        return ret_data


if __name__ == "__main__":
    argss = _parse_args()
    _run(argss.input_path, argss.output_path)
