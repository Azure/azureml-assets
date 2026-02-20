# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Prompt Crafter runner."""

from typing import Optional
import json
import os
import random
import mlflow
import tqdm

from azureml._common._error_definition.azureml_error import AzureMLError

from .checksum import SHA256Checksum
from .prompt_factory import PromptFactory
from aml_benchmark.utils.io import resolve_io_path, read_jsonl_files
from aml_benchmark.utils.logging import get_logger
from aml_benchmark.utils.exceptions import BenchmarkValidationException
from aml_benchmark.utils.error_definitions import BenchmarkValidationError

logger = get_logger(__name__)


class _MLFlowLogger():
    def __init__(self):
        self.steps = 0
        self._max_prompt_word_count = 0
        self._min_prompt_word_count = 100_000
        self._prompt_length_sum = 0

    def increment_step(self):
        self.steps += 1

    def log_prompt_length(self, prompt_length: int):
        self._max_prompt_word_count = max(self._max_prompt_word_count, prompt_length)
        self._min_prompt_word_count = min(self._min_prompt_word_count, prompt_length)
        self._prompt_length_sum += prompt_length

    def log_aggregates(self):
        mlflow.log_metric("max_prompt_word_count", self._max_prompt_word_count)
        mlflow.log_metric("min_prompt_word_count", self._min_prompt_word_count)
        mlflow.log_metric("average_prompt_word_count", self._prompt_length_sum / self.steps)
        mlflow.log_metric("total_prompts", self.steps)

    def save_parameters(self, params, output_mltable):
        if output_mltable is not None:
            try:
                mlflow.log_dict(params, "params.json")

                with open(os.path.join(output_mltable, "params.json"), "w") as f:
                    json.dump(params, f)
            except Exception as ex:
                logger.warning(
                    f"Failed to save parameters to mlflow folder {output_mltable} due to {ex}")


class PromptCrafter:
    """Prompt Crafter class to create prompts from input data."""

    OUTPUT_FILENAME = "few_shot_prompt.jsonl"
    MLTABLE_FILENAME = "MLTable"

    def __init__(
        self,
        test_data: str,
        prompt_type: str,
        n_shots: int,
        random_seed: int,
        output_file: str,
        output_pattern: Optional[str],
        metadata_keys: Optional[str],
        prompt_pattern: Optional[str],
        few_shot_pattern: Optional[str],
        few_shot_separator: Optional[str],
        few_shot_data: Optional[str],
        prefix: Optional[str],
        label_map: Optional[str],
        additional_payload: Optional[str],
        system_message: Optional[str],
        base_prompt_factory_cls: Optional[PromptFactory] = PromptFactory,
        output_mltable: Optional[str] = None,
        ground_truth_column_name: Optional[str] = None,
        additional_columns: Optional[str] = None,
    ):
        """Initialize the prompt crafter."""
        self.metadata_keys = metadata_keys
        self.additional_payload = additional_payload
        self.ground_truth_column_name = ground_truth_column_name
        self.additional_columns = additional_columns
        params = {k: v for k, v in locals().items() if k not in ["self", "base_prompt_factory_cls", "params"]}
        self.mlflow_logger = _MLFlowLogger()
        self.mlflow_logger.save_parameters(params=params, output_mltable=output_mltable)

        # set seed globally
        random.seed(random_seed)

        # sample few shot
        few_shot_pool = self._read_few_shot_pool(few_shot_file=few_shot_data)

        # create input/output file
        input_paths = resolve_io_path(test_data)
        if len(input_paths) == 1:
            self.input_path = input_paths[0]
        else:
            mssg = f"Multiple or No input files found for {test_data}"
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )

        self.output_path = output_file
        self.mltable_output_path = self._prepare_ml_table_output_file(output_mltable=output_mltable)

        # Init prompt factory
        typed_prompt_factory_cls = base_prompt_factory_cls.from_type(prompt_type)
        self.prompt_factory = typed_prompt_factory_cls(
            n_shots=n_shots,
            prompt_pattern=prompt_pattern,
            few_shot_pattern=few_shot_pattern,
            few_shot_pool=few_shot_pool,
            few_shot_separator=few_shot_separator,
            prefix=prefix,
            ground_truth_column_name=ground_truth_column_name,
            additional_columns=additional_columns,
            label_map_str=label_map,
            output_pattern=output_pattern,
            system_message=system_message,
            metadata_keys=metadata_keys,
            additional_payload=additional_payload
        )

    @staticmethod
    def _prepare_ml_table_output_file(output_mltable: str):
        """Prepare the mltable output file."""
        if output_mltable is None:
            return None
        mltable_output_path = os.path.join(output_mltable, PromptCrafter.OUTPUT_FILENAME)
        # must create MLTable file for the mltable output
        mltable_file_output_path = os.path.join(output_mltable, PromptCrafter.MLTABLE_FILENAME)
        s = """type: mltable
paths:
  - pattern: ./*.jsonl
transformations:
  - read_json_lines:
      encoding: utf8
      include_path_column: false"""
        try:
            with open(mltable_file_output_path, 'w') as f:
                f.write(s)
        except Exception as ex:
            logger.warning(
                f"Failed to prepare mltable file {output_mltable} as {ex}")
            return None
        return mltable_output_path

    @staticmethod
    def _read_few_shot_pool(few_shot_file: str = None):
        few_shot_pool = None
        if few_shot_file is not None:
            # create input/output file
            few_shot_paths = resolve_io_path(few_shot_file)
            if len(few_shot_paths) == 1:
                few_shot_pool = read_jsonl_files(few_shot_paths)
            else:
                mssg = f"Multiple or No input files found for {few_shot_file}"
                raise BenchmarkValidationException._with_error(
                    AzureMLError.create(BenchmarkValidationError, error_details=mssg)
                )

        return few_shot_pool

    def row_output_post_process(self, row_output_data):
        """Post process the output data for the row."""
        batch_request_metadata = {}
        if self.metadata_keys:
            for k in self.metadata_keys.split(","):
                k = k.strip()
                if k in row_output_data:
                    batch_request_metadata[k] = row_output_data.pop(k)
        # completion is pass through in OAI API component but to be consumable
        # by batch scoring component, we need to add it to the metadata
        completion = row_output_data.pop("completion", None)
        if completion:
            batch_request_metadata["completion"] = completion
        new_data = {
            **row_output_data,
            "_batch_request_metadata": batch_request_metadata,
        }
        return new_data

    def run(self):
        """Create prompts by iterating over the input files."""
        checksum = SHA256Checksum()

        if self.mltable_output_path is not None:
            f_mltable = open(self.mltable_output_path, "w")

        with open(self.output_path, "w") as f:
            with open(self.input_path) as input_f:
                for index, line in enumerate(tqdm.tqdm(input_f)):
                    self.mlflow_logger.increment_step()
                    row = json.loads(line)

                    output_data = self.prompt_factory.process_row(row=row, index=index)

                    self.mlflow_logger.log_prompt_length(prompt_length=output_data.get("prompt_length", 0))
                    checksum.update(output_data)
                    f.write(json.dumps(output_data) + "\n")

                    if self.mltable_output_path is not None:
                        ml_table_output_data = self.row_output_post_process(output_data)
                        f_mltable.write(json.dumps(ml_table_output_data) + "\n")

        # closing the mltable output file (if exists)
        if self.mltable_output_path is not None:
            f_mltable.close()

        self.mlflow_logger.log_aggregates()

        # NOTE: currently, openai_api component handles the payload parameters
        # like temperature etc. This is why we compute the checksum on output_data
        # instead of new_data. Conceptually, prompt crafter should be preparing the payloads
        # so we should include them into the checksum calculation.
        output_data_checksum = checksum.digest()
        mlflow.log_text(output_data_checksum, "checksum.txt")

        return output_data_checksum
