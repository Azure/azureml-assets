# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Inference Postprocessor class and runner methods for 3P."""

import json
import os
import re
import jinja2
import codecs
import numpy as np
import pandas as pd
from collections import OrderedDict
from typing import Union, List

from azureml._common._error_definition.azureml_error import AzureMLError
from utils.error_definitions import BenchmarkValidationError, BenchmarkUserError
from utils.exceptions import BenchmarkValidationException, BenchmarkUserException
from utils.logging import get_logger
from utils.io import resolve_io_path, read_jsonl_files

logger = get_logger(__name__)

jinja2.filters.FILTERS['zip'] = zip
ENV = jinja2.Environment()
ENV.globals.update(zip=zip)

def get_prompt(data: dict, remove_prompt_prefix: bool = True):
    return data.get("prompt") if remove_prompt_prefix else None

class InferencePostprocessor(object):
    """Inference Postprocessor object class."""

    def __init__(
        self,
        prediction_dataset: str = None,
        prediction_column_name: str = None,
        ground_truth_dataset: str = None,
        ground_truth_column_name: str = None,
        pred_probs_dataset: str = None,
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
        **kwargs
    ) -> None:
        """Inference Postprocessor class.

        :param prediction_dataset: Path to the jsonl file to load the prediction dataset.
        :param prediction_column_name: Name of prediction column/key.
        :param ground_truth_dataset: Path to the jsonl file to load the prediction dataset.
        :param ground_truth_column_name: Name of ground truth column/key.
        :param pred_probs_dataset: Path to the jsonl file to load the prediction probabilities dataset.
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
        :param remove_prompt_prefix: A boolean flag, when set to True, remove the prompt generated by prompt crafter \
            if the generated text contains it when flag in model's parameter return_full_text is set to True.
        :param template: Jinja template containing the extraction logic of inference post-processing.
        :param script_path: Path to the custom preprocessor python script provided by user.
        :param output_dataset: Path to the jsonl file where the processed data will be saved.
        :param prediction_dir: Path to the directory containing the jsonl file with the inference results. If \
            prediction_dataset is specified, prediction_dataset takes priority.
        :param prediction_filename: The name of the jsonl file with the inference results. If \
            prediction_dataset is specified, prediction_dataset takes priority.
            The name of the jsonl file with the inference results. Supports any glob pattern that returns a unique .jsonl \
            file within the specified directory. Gets ignored if prediction_dataset is specified.
        :return: None
        """
        self.prediction_dataset = prediction_dataset
        self.prediction_column_name = prediction_column_name
        self.ground_truth_dataset = ground_truth_dataset
        self.ground_truth_column_name = ground_truth_column_name
        self.pred_probs_dataset = pred_probs_dataset
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
            mssg = (
                "Path to load the prediction dataset is not provided."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if len([
            file for file in resolve_io_path(self.prediction_dataset) if file.endswith(".jsonl")
        ]) == 0:
            mssg = "No .jsonl files found in the given prediction dataset."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.prediction_column_name is None:
            mssg = (
                "Prediction column name is not provided."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        if self.user_postprocessor and not self.user_postprocessor.endswith('.py'):
            mssg = (
                "Please provide python script containing your custom postprocessor logic."
            )
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )

    def read_ground_truth_dataset(self, result_df) -> pd.DataFrame:
        """Read the ground truth dataset if provided."""
        if self.ground_truth_dataset:
            actual_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.ground_truth_dataset)))
            if self.ground_truth_column_name:
                result_df[self.ground_truth_column_name] = actual_df[self.ground_truth_column_name]
            else:
                result_df = actual_df
        return result_df

    def read_pred_probs_dataset(self, result_df) -> pd.DataFrame:
        """Read the prediction probabilities dataset if provided."""
        if self.pred_probs_dataset:
            probs_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.pred_probs_dataset)))
            if self.label_map:
                self.label_map = json.loads(self.label_map)
                probs_df.rename(columns=self.label_map, inplace=True)
            probs_df = probs_df.add_prefix('probs_')
            result_df = pd.concat([result_df, probs_df], axis=1)
        return result_df

    def apply_find_first(self, text: str) -> str:
        """Finds first occurence of any candidate in completion."""
        if self.find_first:
            min_index = len(text)
            first_candidate = ""
            candidates = list(map(lambda x: x.strip(), self.find_first.split(',')))
            for candidate in candidates:
                index = text.find(candidate)
                if index != -1 and index < min_index:
                    min_index = index
                    first_candidate = candidate
            return first_candidate
        return text

    def apply_regex_expr(self, text:str) -> str:
        if self.regex_expr:
            self.regex_expr = json.loads(json.dumps(self.regex_expr))
            matches = re.search(self.regex_expr, text, flags=re.DOTALL)
            if matches is None or len(matches.groups()) == 0:
                return text
        return matches.group(1)

    def apply_extract_number(self, text: str,
                             default: str = "0") -> str:
        """Extracts first or last number from completion."""
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
                        parts = m.split(',')
                        if all(len(part) == 3 for part in parts[1:]):
                            m = ''.join(parts)
                        else:
                            m = parts[-1] if strategy == "last" else parts[0]
                    try:
                        # Test that the matched string is a number
                        val = np.fromstring(m, sep=" ")
                        return m
                    except SyntaxError:
                        # we matched with something that is not a number
                        pass
        return default

    def _convert_to_unicode(self, text: str) -> str:
        """Convert from a raw string to a unicode string.

        Example:
            >>> "\nExample".startswith(r"\n") # False
            >>> "\nExample".startswith(codecs.decode(r"\n", "unicode_escape")) # True
        """
        return codecs.decode(text, "unicode_escape")

    def remove_prefix(self, text: str, prefix: str) -> str:
        if text.startswith(prefix):
            text = text[len(prefix):]
        elif (self._convert_to_unicode(text)).startswith(self._convert_to_unicode(prefix)):
            text = text[len(prefix):]
        return text

    def apply_remove_prefixes(self, text: str) -> str:
        if self.remove_prefixes:
            prefixes = self.remove_prefixes.split(",")
            for prefix in prefixes:
                text = self.remove_prefix(text, prefix)
        return text

    def apply_strip_characters(self, text: str) -> str:
        if self.strip_characters:
            text = text.strip(self.strip_characters)
        return text

    def apply_label_map(self, data) -> Union[pd.DataFrame, str]:
        if self.label_map:
            self.label_map = json.loads(self.label_map)
            col_to_encode = self.label_map.get('column_name', None)
            if col_to_encode is None:
                col_to_encode = self.prediction_column_name
            if isinstance(data, pd.DataFrame):
                data[col_to_encode] = data[col_to_encode].map(self.label_map)
            elif isinstance(data, str):
                #data[col_to_encode] = self.encoder_config.get(str(out_dict.get(col_to_encode)))
                data = self.label_map.get(data)
            elif isinstance(data, dict):
                data[col_to_encode] = self.label_map.get(str(data.get(col_to_encode)))
        return data

    def apply_remove_prompt_prefix(self, text: str, data: dict = None) -> str:
        prompt_prefix = get_prompt(data, self.remove_prompt_prefix)
        if prompt_prefix and text.startswith(prompt_prefix):
            text = text[len(prompt_prefix):]
        return text

    def apply_separator(self, text: str):
        if self.separator:
            self.separator = json.loads(json.dumps(self.separator))
            text = text.split(self.separator)[0]
        return text

    # def get_processor_execution_order(self, processor_order: List = None) -> List:
    #     if processor_order is None or len(processor_order) == 0:
    #         processor_order = [
    #             self.remove_prompt_prefix,
    #             self.apply_separator,
    #             self.apply_find_first,
    #             self.apply_regex_expr,
    #             self.remove_prefixes,
    #             self.remove_suffixes,
    #             self.apply_label_map
    #         ]
    #     return processor_order

    def extract_using_template(self, key: str=None) -> None:
        # result_df: pd.DataFrame, key: str = None, processor_order: List = None
        """Postprocessor run using template."""
        result_df = pd.DataFrame()
        result_df = self.read_ground_truth_dataset(result_df)
        # read the predcition dataset
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        if self.prediction_column_name in predicted_data[0].keys():
            key = self.prediction_column_name
        else:
            key = key if key else "0"
        template = self.template
        env = jinja2.Environment()
        jinja_template = env.from_string(template)
        for row in predicted_data:
            if key != self.prediction_column_name:
                row[self.prediction_column_name] = row.get(key)
            predicted = row.get(self.prediction_column_name)
            if isinstance(predicted, list):
                try:
                    out_string = jinja_template.render(predicted)
                    pred_list.append(out_string)
                except Exception as e:
                    error_msg = "dictionary update sequence element"    #"jinja2.exceptions.UndefinedError: 'list object' has no attribute 'split'"
                    if isinstance(e, ValueError) and error_msg in e.args[0]:
                        curr_pred_list = []
                        for i in range(0, len(predicted)):
                            curr_pred = {self.prediction_column_name:predicted[i]}
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
        if isinstance(pred_list[0], list) and len(pred_list[0])>1:
            cols = [f"{self.prediction_column_name}_{i+1}" for i in range(len(pred_list[0]))] 
        else:
            cols = self.prediction_column_name
        # result_df[self.prediction_column_name] = pred_list
        result_df[cols] = pred_list
        result_df = self.read_pred_probs_dataset(result_df)
        # combine the records in one pandas dataframe and write it to the jsonl file.
        result_df.to_json(self.result, lines=True, orient='records')
        return

    # def extract_using_template(self) -> None:
    #     """Postprocessor run using template."""
    #     result_df = pd.DataFrame()
    #     if self.ground_truth_dataset:
    #         actual_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.ground_truth_dataset)))
    #     if self.ground_truth_column_name:
    #         result_df[self.ground_truth_column_name] = actual_df[self.ground_truth_column_name]
    #     else:
    #         result_df = actual_df
    #     predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
    #     pred_list = []
    #     if self.prediction_column_name in predicted_data[0].keys():
    #         key = self.prediction_column_name
    #     else:
    #         key = key if key else "0"
    #     template = self.template
    #     env = jinja2.Environment()
    #     jinja_template = env.from_string(template)
    #     for row in predicted_data:
    #         if key != self.prediction_column_name:
    #             row[self.prediction_coumn_name] = row.get(key)
    #         out_string = jinja_template.render(row)
    #         pred_list.append(out_string)
    #     result_df[self.prediction_coumn_name] = pred_list
    #     result_df = self.read_pred_probs_dataset(result_df)
    #     result_df.to_json(self.result, lines=True, orient='records')
    #     return

    def extract_inferences(self, result_df: pd.DataFrame, 
                           key: str = None, processor_order: List = None):
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        if self.prediction_column_name in predicted_data[0].keys():
            key = self.prediction_column_name
        else:
            key = key if key else "0"
        #processor = self.get_processor_execution_order(processor_order)
        for row in predicted_data:
            predicted = row.get(key)
            if isinstance(predicted, list):
                curr_pred_list = []
                for i in range(0, len(predicted)):
                    out_string = predicted[i]
                    # for func in processor_order:
                    #     func(out_string, row)
                    out_string = self.apply_remove_prompt_prefix(out_string, row)
                    out_string = self.apply_remove_prefixes(out_string)
                    out_string = self.apply_separator(out_string)
                    out_string = self.apply_find_first(out_string)
                    out_string = self.apply_extract_number(out_string)
                    out_string = self.apply_regex_expr(out_string)
                    out_string = self.apply_strip_characters(out_string)
                    out_string = self.apply_label_map(out_string)
                    curr_pred_list.append(out_string)
                pred_list.append(curr_pred_list)
            else:
                out_string = predicted
                out_string = self.apply_remove_prompt_prefix(out_string, row)
                out_string = self.apply_remove_prefixes(out_string)
                out_string = self.apply_separator(out_string)
                out_string = self.apply_find_first(out_string)
                out_string = self.apply_extract_number(out_string)
                out_string = self.apply_regex_expr(out_string)
                out_string = self.apply_strip_characters(out_string)
                out_string = self.apply_label_map(out_string)
                pred_list.append(out_string)
        if isinstance(pred_list[0], list) and len(pred_list[0])>1:
            cols = [f"{self.prediction_column_name}_{i+1}" for i in range(len(pred_list[0]))] 
        else:
            cols = self.prediction_column_name
        #result_df[self.prediction_column_name] = pred_list
        result_df[cols] = pred_list
        return result_df

    def run(self):# key: str = None, processor_order: List = None) -> None:
        """Postprocessor runner."""
        if self.user_postprocessor:
            self.run_user_postprocessor()
            return
        # 3P post processor logic is written with the assumption that {tokenizer_config:{return_full_text:False}}
        # is the default setting in model prediction component for text generation models.
        # 1P logic should work as long as they contain "prompt" key and it's associated value in their prediction dataset.
        # Removing completion_key that Babel has, but this can be passed in the kwargs.
        if self.template:
            # process extraction logic based on template
            self.extract_using_template()
            return

        # generic post processor logic apply the parameters in the following order:
        # remove_prompt_prefix, separator, find_first, extract_number, replacement,
        # regex_expr, remove_prefixes, strip_suffixes, label_map
        key = self.kwargs.get("completion_key") if "completion_key" in self.kwargs else None
        processor_order = self.kwargs.get("processor_order") if "processor_order" in self.kwargs else None
        result_df = pd.DataFrame()
        result_df = self.read_ground_truth_dataset(result_df)
        result_df = self.extract_inferences(result_df, key, processor_order)
        result_df = self.read_pred_probs_dataset(result_df)
        result_df.to_json(self.result, lines=True, orient='records')
        return

    def run_user_preprocessor(self) -> None:
        """Postprocessor run using custom template."""
        try:
            os.system(
                f'python {self.user_preprocessor} --prediction_dataset {self.prediction_dataset} \
                --ground_truth_dataset {self.ground_truth_dataset} --output_dataset {self.result}'
            )
        except Exception as e:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(BenchmarkUserError, error_details=e)
            )
