# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Inference Postprocessor class and runner methods for 3P."""

import pandas as pd
import json
import os
import re
import jinja2

from azureml._common._error_definition.azureml_error import AzureMLError
from utils.error_definitions import BenchmarkValidationError, BenchmarkUserError
from utils.exceptions import BenchmarkValidationException, BenchmarkUserException
from utils.logging import get_logger
from utils.io import resolve_io_path, read_jsonl_files

logger = get_logger(__name__)

jinja2.filters.FILTERS['zip'] = zip
ENV = jinja2.Environment()
ENV.globals.update(zip=zip)


class InferencePostprocessor(object):
    """Inference Postprocessor object class."""

    def __init__(
        self,
        prediction_dataset: str = None,
        Y: str = None,
        ground_truth_dataset: str = None,
        y: str = None,
        pred_probs_dataset: str = None,
        encoder_config: str = None,
        separator: str = None,
        regex_expr: str = None,
        extract_value_at_index: int = None,
        strip_prefix: str = None,
        strip_suffix: str = None,
        template: str = None,
        user_postprocessor: str = None,
        output_dataset: str = None
    ) -> None:
        """Inference Postprocessor class.

        :param prediction_dataset: Path to the jsonl file to load the prediction dataset.
        :param Y: Name of prediction column/key.
        :param ground_truth_dataset: Path to the jsonl file to load the prediction dataset.
        :param y: Name of ground truth column/key.
        :param pred_probs_dataset: Path to the jsonl file to load the prediction probabilities dataset.
        :param encoder_config: JSON serialized dictionary to perform mapping. Must contain key-value pair \
            "column_name": "<actual_column_name>" whose value needs mapping, followed by key-value pairs containing \
            idtolabel or labeltoid mappers. Example format: \
            {"column_name":"label", "0":"NEUTRAL", "1":"ENTAILMENT", "2":"CONTRADICTION"}. This is not applicable to \
            custom scripts.
        :param separator: Few shot separator used in prompt crafter.
        :param regex_expr: A regex pattern to extract the answer from the inference results.
        :param extract_value_at_index: The matched regex pattern value to be extracted in case
            multiple strings are found using the pattern provided in parameter `regex_expr`.
        :param strip_prefix: Characters to remove from the beginning of the extracted answer.
        :param strip_suffix: "Characters to remove from the end of the extracted answer."
        :param template: Jinja template containing the extraction logic of inference post-processing.
        :param script_path: Path to the custom preprocessor python script provided by user.
        :param output_dataset: Path to the jsonl file where the processed data will be saved.
        :return: None
        """
        self.prediction_dataset = prediction_dataset
        self.Y = Y
        self.ground_truth_dataset = ground_truth_dataset
        self.y = y
        self.pred_probs_dataset = pred_probs_dataset
        self.encoder_config = encoder_config
        self.separator = separator
        self.regex_expr = regex_expr
        self.index = extract_value_at_index if extract_value_at_index else 0
        self.strip_prefix = strip_prefix
        self.strip_suffix = strip_suffix
        self.template = template
        self.user_postprocessor = user_postprocessor
        self.result = output_dataset
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
        if self.Y is None:
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

    def extract_using_template(self) -> None:
        """Postprocessor run using template."""
        result_df = pd.DataFrame()
        if self.ground_truth_dataset:
            actual_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.ground_truth_dataset)))
        if self.y:
            result_df[self.y] = actual_df[self.y]
        else:
            result_df = actual_df
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        if self.Y in predicted_data[0].keys():
            key = self.Y
        else:
            key = "0"
        template = self.template
        env = jinja2.Environment()
        jinja_template = env.from_string(template)
        for row in predicted_data:
            if key == "0":
                row[self.Y] = row.get(key)
            out_string = jinja_template.render(row)
            pred_list.append(out_string)
        result_df[self.Y] = pred_list
        if self.pred_probs_dataset:
            probs_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.pred_probs_dataset)))
            if self.encoder_config:
                self.encoder_config = json.loads(self.encoder_config)
                probs_df = probs_df.rename(columns=self.encoder_config).add_prefix('probs_')
                result_df = pd.concat([result_df, probs_df], axis=1)
        result_df.to_json(self.result, lines=True, orient='records')
        return

    def run(self) -> None:
        """Postprocessor runner."""
        if self.user_postprocessor:
            self.run_user_postprocessor()
            return
        # this is written with the assumption that {tokenizer_config:{return_full_text:False}} is the
        # default setting in model prediction component for text generation models.
        if self.template:
            # process extraction logic based on template
            self.extract_using_template()
            return
        # generic post processor
        result_df = pd.DataFrame()
        if self.ground_truth_dataset:
            actual_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.ground_truth_dataset)))
            if self.y:
                result_df[self.y] = actual_df[self.y]
            else:
                result_df = actual_df
        predicted_data = read_jsonl_files(resolve_io_path(self.prediction_dataset))
        pred_list = []
        if self.Y in predicted_data[0].keys():
            key = self.Y
        else:
            key = "0"
        if self.separator:
            sep = json.dumps(self.separator)
            separator = json.loads(sep)
        if self.regex_expr:
            self.regex_expr = json.loads(json.dumps(self.regex_expr))
        for row in predicted_data:
            out_string = row.get(key)
            if self.separator:
                out_string = out_string.split(separator)[0]
            if self.regex_expr:
                out_string = re.findall(self.regex_expr, out_string, flags=re.DOTALL)
                if isinstance(out_string[0], tuple):
                    out_string = out_string[0]
                out_string = out_string[self.index]
            if self.strip_prefix:
                out_string = out_string.lstrip(self.strip_prefix)
            if self.strip_suffix:
                out_string = out_string.rstrip(self.strip_suffix)
            pred_list.append(out_string)
        result_df[self.Y] = pred_list
        if self.pred_probs_dataset:
            probs_df = pd.json_normalize(read_jsonl_files(resolve_io_path(self.pred_probs_dataset)))
            if self.encoder_config:
                self.encoder_config = json.loads(self.encoder_config)
                probs_df = probs_df.rename(columns=self.encoder_config).add_prefix('probs_')
                result_df = pd.concat([result_df, probs_df], axis=1)
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
