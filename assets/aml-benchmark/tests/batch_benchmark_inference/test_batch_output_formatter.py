# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Compute Performance Metrics Component."""

from typing import Optional
import json
import os
import uuid
import pytest

from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from ..test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    ModelType,
    PromptType,
)


class TestBatchOutputFormatterComponent:
    """Component test for batch output formatter."""

    EXP_NAME = "batch-output-formatter-test"

    @pytest.mark.parametrize('model_type, prompt_type, use_tiktoken', [
        (ModelType.OAI, PromptType.CHAT_COMPLETION, False),
        (ModelType.OAI, PromptType.CHAT_COMPLETION, True),
        (ModelType.OAI, PromptType.TEXT_GENERATION, True),
        (ModelType.OSS, PromptType.CHAT_COMPLETION, True),
        (ModelType.OSS, PromptType.TEXT_GENERATION, True),
        (ModelType.VISION_OSS, None, False)
    ])
    def test_batch_output_formatter(
        self,
        model_type: Optional[ModelType],
        prompt_type: Optional[PromptType],
        use_tiktoken: bool, temp_dir: str
    ):
        """Test method for batch inference preparer."""
        ml_client = get_mlclient()
        pipeline_job = self._get_pipeline_job(
            self.test_batch_output_formatter.__name__,
            'label',
            'http://test-endpoint.com',
            'question',
            temp_dir,
            model_type,
            prompt_type,
            use_tiktoken,
        )
        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)

        out_dir = os.path.join(temp_dir, "output")
        os.makedirs(out_dir, exist_ok=True)
        self._verify_output(
            pipeline_job, output_dir=out_dir, use_tiktoken=use_tiktoken
        )

    def _get_pipeline_job(
            self,
            display_name: str,
            label_key: str,
            endpoint_url: str,
            additional_columns: Optional[str] = None,
            temp_dir: Optional[str] = None,
            model_type: Optional[ModelType] = None,
            prompt_type: Optional[PromptType] = None,
            use_tiktoken: bool = False,
    ) -> Job:
        pipeline_job = load_yaml_pipeline("batch_output_formatter.yaml")

        # avoid blob exists error when running pytest with multiple workers
        if temp_dir is not None:
            file_path = os.path.join(temp_dir, uuid.uuid4().hex + ".jsonl")
            input_file = Constants.BATCH_OUTPUT_FORMATTER_OSS_TEXT_FILE_PATH
            if model_type == ModelType.OAI:
                if prompt_type == PromptType.CHAT_COMPLETION:
                    input_file = Constants.BATCH_OUTPUT_FORMATTER_OAI_CHAT_FILE_PATH
                elif prompt_type == PromptType.TEXT_GENERATION:
                    input_file = Constants.BATCH_OUTPUT_FORMATTER_OAI_TEXT_FILE_PATH
            elif model_type == ModelType.OSS:
                if prompt_type == PromptType.CHAT_COMPLETION:
                    input_file = Constants.BATCH_OUTPUT_FORMATTER_OSS_CHAT_FILE_PATH
                elif prompt_type == PromptType.TEXT_GENERATION:
                    input_file = Constants.BATCH_OUTPUT_FORMATTER_OSS_TEXT_FILE_PATH
            with open(input_file, "r") as f:
                with open(file_path, "w") as f2:
                    f2.write(f.read())

        # set the pipeline inputs
        if model_type:
            pipeline_job.jobs['run_batch_output_formatter'].inputs.model_type = model_type.value
        pipeline_job.inputs.batch_inference_output = Input(
            type="uri_folder", path=temp_dir
        )
        pipeline_job.inputs.label_column_name = label_key
        pipeline_job.inputs.endpoint_url = endpoint_url
        pipeline_job.inputs.use_tiktoken = use_tiktoken

        pipeline_job.display_name = display_name
        pipeline_job.name = str(uuid.uuid4())

        return pipeline_job

    def _read_data(self, file_path):
        with open(file_path) as f:
            output_records = [json.loads(line) for line in f]
        return output_records

    def _check_columns_in_dfs(self, column_list, dfs):
        for df in dfs:
            for col in column_list:
                assert col in df, f"{col} not found in df."

    def _check_output_data(self, output_dir, file_name, expected_col_list):
        output_file = os.path.join(output_dir, file_name)
        assert os.path.isfile(output_file), f"{output_file} is not a file"
        # Read the output file
        dfs = self._read_data(output_file)
        self._check_columns_in_dfs(expected_col_list, dfs)

    def _verify_output(self, job, output_dir, use_tiktoken: bool):
        prediction_data = job.outputs.predictions.port_name
        perf_data = job.outputs.performance_metadata.port_name
        ground_truth_data = job.outputs.ground_truth.port_name
        for output_name in [prediction_data, perf_data, ground_truth_data]:
            download_outputs(
                job_name=job.name, output_name=output_name, download_path=output_dir
            )
        output_dir = os.path.join(output_dir, "named-outputs")
        self._check_output_data(
            os.path.join(output_dir, "predictions"), "predictions.jsonl", ["prediction"])
        perf_cols = ["start_time_iso", "end_time_iso", "time_taken_ms"]
        if use_tiktoken:
            perf_cols.extend(["input_token_count", "output_token_count"])
        self._check_output_data(
            os.path.join(output_dir, "performance_metadata"),
            "performance_metadata.jsonl", perf_cols
        )
        self._check_output_data(
            os.path.join(
                output_dir, "ground_truth"), "ground_truth.jsonl", ["label"])
