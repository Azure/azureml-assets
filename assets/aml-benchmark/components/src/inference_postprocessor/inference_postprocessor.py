# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Inference Postprocessor class and runner methods for 3P."""

import pandas as pd
import json
import os
import re
import sys
import jinja2
from utils.error_definitions import BenchmarkValidationError
from utils.exceptions import BenchmarkValidationException
from utils.logging import get_logger
from utils.io import read_jsonl_files
from azureml._common._error_definition.azureml_error import AzureMLError

current_folder = os.path.dirname(os.path.abspath(__file__))
parent_folder = os.path.dirname(current_folder)
sys.path.append(parent_folder)
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
        separator: str = None,
        regex_expr: str = None,
        strip_prefix: str = None,
        strip_suffix: str = None,
        template: str = None,
        user_postprocessor: str = None,
        output_dataset: str = None
    ) -> None:
        """Inference Postprocessor class.

        :param prediction_dataset: Path to the directory to load the prediction dataset.
        :param Y: Name of prediction column/key.
        :param ground_truth_dataset: Path to the directory to load the prediction dataset.
        :param y: Name of ground truth column/key.
        :param separator: Few shot separator used in prompt crafter.
        :param regex_expr: A regex pattern to extract the answer from the inference results.
        :param strip_prefix: Characters to remove from the beginning of the extracted answer.
        :param strip_suffix: "Characters to remove from the end of the extracted answer."
        :param template: Jinja template containing the extraction logic of inference post-processing.
        :param script_path: Path to the custom preprocessor python script provided by user.
        :param output_dataset: Path to the dump the processed .jsonl file.
        :return: None
        """
        self.prediction_dataset = prediction_dataset
        self.Y = Y
        self.ground_truth_dataset = ground_truth_dataset
        self.y = y
        self.separator = separator
        self.regex_expr = regex_expr
        self.strip_prefix = strip_prefix
        self.strip_suffix = strip_suffix
        self.template = template
        self.user_postprocessor = user_postprocessor
        self.result = output_dataset

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
            actual_df = pd.json_normalize(read_jsonl_files([self.ground_truth_dataset]))
        if self.y:
            result_df[self.y] = actual_df[self.y]
        else:
            result_df = actual_df
        predicted_data = read_jsonl_files([self.prediction_dataset])
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
            actual_df = pd.json_normalize(read_jsonl_files([self.ground_truth_dataset]))
        if self.y:
            result_df[self.y] = actual_df[self.y]
        else:
            result_df = actual_df
        predicted_data = read_jsonl_files([self.prediction_dataset])
        pred_list = []
        if self.Y in predicted_data[0].keys():
            key = self.Y
        else:
            key = "0"
        if self.separator:
            sep = json.dumps(self.separator)
            # sep = self.separator
            separator = json.loads(sep)
        if self.regex_expr:
            self.regex_expr = json.loads(json.dumps(self.regex_expr))
            self.regex_expr = re.compile(f"{self.regex_expr}")
        for row in predicted_data:
            out_string = row.get(key)
            if self.separator:
                out_string = out_string.split(separator)[0]
            if self.regex_expr:
                out_string = re.findall(self.regex_expr, out_string)[-1]
            if self.strip_prefix:
                out_string = out_string.lstrip(self.strip_prefix)
            if self.strip_suffix:
                out_string = out_string.rstrip(self.strip_suffix)
            pred_list.append(out_string)
        result_df[self.Y] = pred_list
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
            logger.error('Script failed', e)
        return
