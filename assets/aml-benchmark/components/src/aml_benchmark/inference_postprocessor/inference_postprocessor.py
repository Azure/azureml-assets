# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Inference Postprocessor class and runner methods for 3P."""

import json
import re
import jinja2
import codecs
import subprocess
import numpy as np
import pandas as pd
import random

from copy import deepcopy
from typing import List, Optional, Union

from azureml._common._error_definition.azureml_error import AzureMLError
from aml_benchmark.utils.error_definitions import (
    BenchmarkValidationError,
    BenchmarkUserError,
)
from aml_benchmark.utils.exceptions import (
    BenchmarkValidationException,
    BenchmarkUserException,
)
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files

logger = get_logger(__name__)

jinja2.filters.FILTERS["zip"] = zip
ENV = jinja2.Environment()
ENV.globals.update(zip=zip)


# Adapted from the MMMU codebase.
def extract_choice_from_response(response: str, choices: List[str], choose_randomly_if_no_match: bool=True) -> Optional[int]:
    response = (" " + response.strip(",.!?;:'") + " ").lower()

    matching_choice_indexes, response_start_indexes = [], []
    for i, choice in enumerate(choices):
        j = response.rfind(choice.strip().lower())
        if j != -1:
            matching_choice_indexes.append(i)
            response_start_indexes.append(j)

    if len(matching_choice_indexes) == 0:
        if choose_randomly_if_no_match:
            return random.randint(0, len(choices) - 1)
        return None

    if len(matching_choice_indexes) == 1:
        return matching_choice_indexes[0]

    i = response_start_indexes.index(max(response_start_indexes))
    return [matching_choice_indexes[i]]


def fit_response_to_label(response, choices, label):
    if len(choices) == 0:
        choice_index = extract_choice_from_response(response, label.split("||"), choose_randomly_if_no_match=False)
        if choice_index is not None:
            response = label
        else:
            response = ""
    else:
        choice_index = extract_choice_from_response(response, choices)
        response = chr(ord("A") + choice_index)

    return response


def get_prompt(data: dict, remove_prompt_prefix: bool = True):
    """Return the prompt prefix if 'prompt' keyword is present in the data."""
    return data.get("prompt") if remove_prompt_prefix else None


class InferencePostprocessor(object):
    """Inference Postprocessor object class."""

    def __init__(
        self,
        prediction_dataset: str = None,
        prediction_column_name: str = None,
        ground_truth_dataset: str = None,
        ground_truth_column_name: str = None,
        additional_columns: str = None,
        separator: str = None,
        find_first: str = None,
        regex_expr: str = None,
        remove_prefixes: str = None,
        strip_characters: str = None,
        label_map: str = None,
        template: str = None,
        user_postprocessor: str = None,
        output_dataset: str = None,
        extract_number: str = None,
        remove_prompt_prefix: str = False,
        prediction_dir: str = None,
        prediction_filename: str = "few_shot_prompt*",
        **kwargs,
    ) -> None:
        """Inference Postprocessor class.

        :param prediction_dataset: Path to the jsonl file to load the prediction dataset.
        :param prediction_column_name: Name of prediction column/key.
        :param ground_truth_dataset: Path to the jsonl file to load the prediction dataset.
        :param ground_truth_column_name: Name of ground truth column/key.
        :param additional_columns: Name of additional columns which may be useful for metric computation.
        :param separator: Few shot separator used in prompt crafter.
        :param find_first: A list of strings to search for in the inference results. The first occurrence \
            of each string will be extracted. Must provide a comma-separated list of strings.
            Example, for the following input:
            >>> find_first = "positive,negative"
            >>> completion = "This is a positive example, not negative"
            # Output: "positive"
        :param regex_expr: A regex pattern to extract the answer from the inference results.
        :param remove_prefixes: A set of string prefixes separated by comma list of string prefixes to be removed \
            from the inference results in sequence. This can also be used to remove the prompt from the inference \
            results. The prefixes should be separated by a comma.
        :param strip_characters: A set of characters to remove from the beginning or end of the extracted answer.\
            It is applied in the very end of the extraction process.
        :param label_map: JSON serialized dictionary to perform mapping. Must contain key-value pair \
            "column_name": "<actual_column_name>" whose value needs mapping, followed by key-value pairs containing \
            idtolabel or labeltoid mappers. Example format: \
            {"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}. This is not applicable to \
            custom scripts.
        :param extract_number: A enum that takes two values - "first" or "last". The default value is "first". \
            If the inference results contain a number, this can be used to extract the first or last number in the \
            inference results. The number will be extracted as a string.
            Example:
            >>> extract_number = "first"
            >>> completion = "Adding 0.3 to 1,000 gives 1,000.3"
            # Output: "0.3"
            Example:
            >>> extract_number = "last"
            >>> completion = "Adding 0.3 to 1,000 gives 1,000.3"
            # Output: "1000.3"
        :param remove_prompt_prefix: A boolean flag, when set to True, remove the prompt generated by prompt \
            crafter if the generated text contains it when flag in model's parameter return_full_text is set to True.
        :param template: Jinja template containing the extraction logic of inference post-processing.
        :param script_path: Path to the custom preprocessor python script provided by user.
        :param output_dataset: Path to the jsonl file where the processed data will be saved.
        :param prediction_dir: Path to the directory containing the jsonl file with the inference results. If \
            prediction_dataset is specified, prediction_dataset takes priority.
        :param prediction_filename: The name of the jsonl file with the inference results. If \
            prediction_dataset is specified, prediction_dataset takes priority.
            The name of the jsonl file with the inference results. Supports any glob pattern that returns a \
            unique .jsonl file within the specified directory. Gets ignored if prediction_dataset is specified.
        :return: None
        """
        self.prediction_dataset = prediction_dataset
        self.prediction_column_name = prediction_column_name
        self.ground_truth_dataset = ground_truth_dataset
        self.ground_truth_column_name = ground_truth_column_name
        self.additional_columns = additional_columns
        self.label_map = label_map
        self.separator = separator
        self.regex_expr = regex_expr
        self.remove_prefixes = remove_prefixes
        self.strip_characters = strip_characters
        self.remove_prompt_prefix = remove_prompt_prefix
        self.find_first = find_first
        self.extract_number = extract_number
        self.template = template
        self.user_postprocessor = user_postprocessor
        self.result = output_dataset
        self.prediction_dir = prediction_dir
        self.prediction_filename = prediction_filename
        self.kwargs = kwargs
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post init call."""
        self.validate()

    def validate(self) -> None:
        """Validate the parameters."""
        if self.prediction_dataset is None:
            mssg = "Path to load the prediction dataset is not provided."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.prediction_column_name is None:
            mssg = "Prediction column name is not provided."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.user_postprocessor and not self.user_postprocessor.endswith(".py"):
            mssg = "Please provide python script containing your custom postprocessor logic."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )

    def read_ground_truth_dataset(self) -> pd.DataFrame:
        """
        Read the ground truth dataset if provided.

        If ground truth dataset is n-D array, then read only the provided ground truth column name
        and the additional columns.
        """
        result_df = pd.DataFrame()
        if self.ground_truth_dataset:
            actual_df = pd.json_normalize(
                read_jsonl_files(resolve_io_path(self.ground_truth_dataset))
            )
            if not self.ground_truth_dataset and not self.additional_columns:
                return actual_df
            if self.ground_truth_column_name:
                result_df[self.ground_truth_column_name] = actual_df[
                    self.ground_truth_column_name
                ]
            if self.additional_columns:
                columns = [col.strip() for col in self.additional_columns.split(",")]
                if "" in columns:
                    logger.warning(
                        "Received a column name as '' in additional_fields. "
                        "Please check if extra comma is provided between two column names. "
                        "Dropping columns named as '' from additional_fields input."
                    )
                    columns = [
                        col for col in columns if col
                    ]  # and col in actual_df.columns.tolist()]
                missing_columns = [
                    col for col in columns if col not in actual_df.columns.tolist()
                ]
                if len(missing_columns) > 0:
                    raise BenchmarkUserException._with_error(
                        AzureMLError.create(
                            BenchmarkUserError,
                            error_details=(
                                f"The columns {missing_columns} provided in the additional_columns field is not "
                                "found in the ground truth dataset. Please make sure that the all columns "
                                "provided in this field is present in the groun_truth dataset."
                            ),
                        )
                    )
                result_df[columns] = actual_df[columns]
        return result_df

    def apply_find_first(self, text: str) -> str:
        """Find and return first occurence of any candidate in the text."""
        if self.find_first:
            min_index = len(text)
            first_candidate = ""
            candidates = list(map(lambda x: x.strip(), self.find_first.split(",")))
            for candidate in candidates:
                index = text.find(candidate)
                if index != -1 and index < min_index:
                    min_index = index
                    first_candidate = candidate
            return first_candidate
        return text

    def apply_regex_expr(self, text: str) -> str:
        """Apply regex on the given text."""
        if self.regex_expr:
            # self.regex_expr = json.loads(json.dumps(self.regex_expr))
            self.regex_expr = self._convert_to_unicode(self.regex_expr)
            matches = re.search(self.regex_expr, text, flags=re.DOTALL)
            if matches is None or len(matches.groups()) == 0:
                return text
            return matches.group(1)
        return text

    def apply_extract_number(self, text: str, default: str = "0") -> str:
        """Extract the first or last number from text if provided."""
        if self.extract_number is None:
            return text
        number_pattern = re.compile(r"(\-?[0-9\.\,\s]+)")
        match = number_pattern.findall(text)
        strategy = self.extract_number
        if match:
            if strategy == "last":
                match = match[::-1]
            for m in match:
                if not re.search(r"\d", m):
                    # we matched with a comma or full-stop, skip this
                    continue
                else:
                    m = m.strip()
                    m = m.rstrip(".")
                    # we only accept space and comma as separators of 3 digits in a number
                    m = m.replace(" ", ",")
                    m = m.strip(",")
                    if "," in m:
                        parts = m.split(",")
                        if all(len(part) == 3 for part in parts[1:]):
                            m = "".join(parts)
                        else:
                            m = parts[-1] if strategy == "last" else parts[0]
                    try:
                        # Test that the matched string is a number
                        np.fromstring(m, sep=" ")
                        return m
                    except SyntaxError:
                        # we matched with something that is not a number
                        pass
        if self.kwargs.get("extract_number_strategy_default_value") is not None:
            default = self.kwargs.get("extract_number_strategy_default_value")
        return default

    def _convert_to_unicode(self, text: str) -> str:
        r"""
        Convert from a raw string to a unicode string.

        Example:
            >>> "\nExample".startswith(r"\n") # False
            >>> "\nExample".startswith(codecs.decode(r"\n", "unicode_escape")) # True
        """
        return codecs.decode(text, "unicode_escape")

    def remove_prefix(self, text: str, prefix: str) -> str:
        """Remove string prefix in the given text."""
        if text.startswith(prefix):
            text = text[len(prefix):]
        elif (self._convert_to_unicode(text)).startswith(
            self._convert_to_unicode(prefix)
        ):
            text = text[len(prefix):]
        return text

    def apply_remove_prefixes(self, text: str) -> str:
        """Remove string prefixes in the given text."""
        if self.remove_prefixes:
            prefixes = self.remove_prefixes.split(",")
            for prefix in prefixes:
                text = self.remove_prefix(text, prefix)
        return text

    def apply_strip_characters(self, text: str) -> str:
        """Remove set of characters from the begining and end the given text."""
        if self.strip_characters:
            text = text.strip(self.strip_characters)
        return text

    def apply_label_map(self, data: str) -> Union[pd.DataFrame, str]:
        """Apply label map on the data.

        :param data: A json serialized dictionary.
        """
        if self.label_map:
            if not isinstance(self.label_map, dict):
                self.label_map = json.loads(self.label_map)
            col_to_encode = self.label_map.get("column_name", None)
            if col_to_encode is None:
                col_to_encode = self.prediction_column_name
            if isinstance(data, pd.DataFrame):
                data[col_to_encode] = data[col_to_encode].map(self.label_map)
            elif isinstance(data, str):
                data = self.label_map.get(data)
            elif isinstance(data, dict):
                data[col_to_encode] = self.label_map.get(str(data.get(col_to_encode)))
        return data

    def apply_remove_prompt_prefix(self, text: str, data: dict = None) -> str:
        """Remove prompts that has been added as prefix in the given text."""
        prompt_prefix = get_prompt(data, self.remove_prompt_prefix)
        if prompt_prefix and text.startswith(prompt_prefix):
            text = text[len(prompt_prefix):]
        return text

    def apply_separator(self, text: str) -> str:
        """Apply few shot separator used in prompt crafter."""
        if self.separator:
            # self.separator = json.loads(json.dumps(self.separator))
            self.separator = self._convert_to_unicode(self.separator)
            text = text.split(self.separator)[0]
        return text

    def run_processor_using_template(self, key: str = None) -> None:
        """Postprocessor run using template."""
        result_df = pd.DataFrame()
        result_df = self.read_ground_truth_dataset()
        # read the prediction dataset
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        if self.prediction_column_name in predicted_data[0].keys():
            key = self.prediction_column_name
        else:
            key = key if key else "0"
        template = self.template
        env = jinja2.Environment()
        jinja_template = env.from_string(template)
        for idx, row in enumerate(predicted_data):
            if key != self.prediction_column_name:
                row[self.prediction_column_name] = row.get(key)
            predicted = row.get(self.prediction_column_name)
            if predicted is None:
                logger.warning(
                    f"Received None as prediction at index {idx}. \
                               Falling back to an empty string."
                )
                pred_list.append("")
            elif isinstance(predicted, list) and len(predicted) == 0:
                logger.warning(
                    f"Received an empty array of predictions at index {idx}. \
                               Falling back to an empty string."
                )
                pred_list.append("")
            elif isinstance(predicted, list):
                try:
                    out_string = jinja_template.render(predicted)
                    pred_list.append(out_string)
                except Exception as e:
                    # "jinja2.exceptions.UndefinedError: 'list object' has no attribute 'split'"
                    error_msg = "dictionary update sequence element"
                    if isinstance(e, ValueError) and error_msg in e.args[0]:
                        curr_pred_list = []
                        for i in range(0, len(predicted)):
                            curr_pred = {self.prediction_column_name: predicted[i]}
                            out_string = jinja_template.render(curr_pred)
                            curr_pred_list.append(out_string)
                        pred_list.append(curr_pred_list)
                    else:
                        raise BenchmarkUserException._with_error(
                            AzureMLError.create(BenchmarkUserError, error_details=e)
                        )
            else:
                out_string = jinja_template.render(row)
                pred_list.append(out_string)
        if isinstance(pred_list[0], list) and len(pred_list[0]) > 1:
            cols = [
                f"{self.prediction_column_name}_{i+1}" for i in range(len(pred_list[0]))
            ]
        else:
            cols = self.prediction_column_name
        result_df[cols] = pred_list
        # combine the records in one pandas dataframe and write it to the jsonl file.
        result_df.to_json(self.result, lines=True, orient="records")
        return

    def apply_generic_processor(self, out_string: str, row: dict = None) -> List:
        """Processor steps."""
        out_string = self.apply_remove_prompt_prefix(out_string, row)
        out_string = self.apply_remove_prefixes(out_string)
        out_string = self.apply_separator(out_string)
        out_string = self.apply_find_first(out_string)
        out_string = self.apply_extract_number(out_string)
        out_string = self.apply_regex_expr(out_string)
        out_string = self.apply_strip_characters(out_string)
        out_string = self.apply_label_map(out_string)
        return out_string

    def extract_multi_choice(self, actual_df):
        print("b1", actual_df.columns.tolist())

        multi_choice_predictions = []

        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        for i, row in enumerate(predicted_data):
            response = row.get(self.prediction_column_name)
            label = actual_df.iloc[i][self.ground_truth_column_name]
            print("b2", i, response, label)

            answer_options = actual_df.iloc[i]["answer_options"]
            if len(answer_options) == 0:
                choices = []
            else:
                choices = answer_options.split("||")
            print("b3", choices)

            p = fit_response_to_label(response, choices, label)
            print("b4", p)

            multi_choice_predictions.append(p)

        return pd.DataFrame(multi_choice_predictions, columns=["predictions"])

    def extract_inferences(
        self, key: str = None, processor_order: List = None
    ) -> pd.DataFrame:
        """Extract inferences using generic method if no template or custom post-processor is provided."""
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        if self.prediction_column_name in predicted_data[0].keys():
            key = self.prediction_column_name
        else:
            key = key if key else "0"
        for idx, row in enumerate(predicted_data):
            predicted = row.get(key)
            if predicted is None:
                logger.warning(
                    f"Received None as prediction at index {idx}. \
                               Falling back to an empty string."
                )
                pred_list.append("")
            elif isinstance(predicted, list) and len(predicted) == 0:
                logger.warning(
                    f"Received an empty array of predictions at index {idx}. \
                               Falling back to an empty string."
                )
                pred_list.append("")
            elif isinstance(predicted, list) and len(predicted[0]) > 1:
                curr_pred_list = [
                    self.apply_generic_processor(out_string, row)
                    for out_string in predicted
                ]
                pred_list.append(curr_pred_list)
            else:
                out_string = predicted if isinstance(predicted, str) else predicted[0]
                pred_list.append(
                    self.apply_generic_processor(out_string, row)
                    if out_string != ""
                    else out_string
                )
        if isinstance(pred_list[0], list) and len(pred_list[0]) > 1:
            cols = [
                f"{self.prediction_column_name}_{i+1}" for i in range(len(pred_list[0]))
            ]
        else:
            cols = [self.prediction_column_name]
        result_df = pd.DataFrame(pred_list, columns=cols)
        return result_df

    def run(self) -> None:
        """Postprocessor runner."""
        if self.user_postprocessor:
            self.run_user_postprocessor()
            return
        # 3P post processor logic is written with the assumption that {tokenizer_config:{return_full_text:False}}
        # is the default setting in model prediction component for text generation models.
        # 1P logic should work as long as they contain "prompt" key and it's associated value in their
        # prediction dataset.
        # Removing completion_key that Babel has, but this can be passed in the kwargs.
        if self.template:
            # process extraction logic based on template
            self.run_processor_using_template()
            return

        # generic post processor logic apply the parameters in the following order:
        # remove_prompt_prefix, separator, find_first, extract_number, replacement,
        # regex_expr, remove_prefixes, strip_suffixes, label_map
        key = (
            self.kwargs.get("completion_key")
            if "completion_key" in self.kwargs
            else None
        )
        processor_order = (
            self.kwargs.get("processor_order")
            if "processor_order" in self.kwargs
            else None
        )
        actual_df = self.read_ground_truth_dataset()
        # predicted_df = self.extract_inferences(key, processor_order)
        predicted_df = self.extract_multi_choice(actual_df)
        # pd.concat([actual_df[self.ground_truth_column_name], predicted_df], axis=1).to_json(
        #     self.result, lines=True, orient="records"
        # )
        predicted_df.to_json(self.result, lines=True, orient="records")
        return

    def __get_parameters(self) -> dict:
        return deepcopy(self.__dict__)

    def run_user_postprocessor(self) -> None:
        """Postprocessor run using custom template."""
        params_dict = deepcopy(self.__get_parameters())
        postprocessor_script = params_dict.pop("user_postprocessor")
        argss = [
            "--prediction_dataset",
            params_dict.pop("prediction_dataset"),
            "--output_dataset",
            params_dict.pop("result"),
        ]
        if self.ground_truth_dataset:
            argss.extend(
                ["--ground_truth_dataset", params_dict.pop("ground_truth_dataset")]
            )
        additional_parameters = json.dumps(params_dict)
        argss.extend(["--additional_parameters", f"'{additional_parameters}'"])
        argss = " ".join(argss)
        try:
            _ = subprocess.check_output(
                f"python {postprocessor_script} {argss}",
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                shell=True,
            )
        except subprocess.CalledProcessError as e:
            error_message = e.output.strip()
            raise BenchmarkUserException._with_error(
                AzureMLError.create(BenchmarkUserError, error_details=error_message)
            )
