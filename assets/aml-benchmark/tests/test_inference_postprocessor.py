# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Test script for Inference Postprocessor Component."""

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
import pytest
import os

from azure.ai.ml.constants import AssetTypes
from utils import (
    load_yaml_pipeline,
    get_mlclient,
    download_outputs,
    assert_logged_params,
)

INPUT_PATH = os.path.join(os.getcwd(), 'data')


class TestInferencePostprocessor:
    """Test Inference Postprocessor."""

    EXP_NAME = "postprocessor-test"

    @pytest.mark.parametrize(
        "prediction_dataset, prediction_column_name, ground_truth_dataset, ground_truth_column_name, separator, \
        regex_expr, strip_prefix, strip_suffix, template, script_path",
        [
            (
                os.path.join(INPUT_PATH, "sample_predictions.jsonl"), "prediction",
                os.path.join(INPUT_PATH, "sample_ground_truths.jsonl"), "final_answer", None, None,
                None, None, '{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}', None
            ),
        ],
    )
    def test_inference_postprocessor_as_component(
        self,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        strip_prefix: str,
        strip_suffix: str,
        template: str,
        script_path: str
    ) -> None:
        """Inference Postprocessor component test."""
        ml_client = get_mlclient()
        exp_name = f"{self.EXP_NAME}"
        pipeline_job = self._get_pipeline_job(
            prediction_dataset,
            prediction_column_name,
            ground_truth_dataset,
            ground_truth_column_name,
            separator,
            regex_expr,
            strip_prefix,
            strip_suffix,
            template,
            script_path,
            "Dataset post-processor pipeline test",
            pipeline_file="inference_postprocessor_pipeline.yaml"
        )
        # submit the pipeline job
        try:
            pipeline_job = ml_client.create_or_update(
                pipeline_job, experiment_name=self.EXP_NAME
            )
            ml_client.jobs.stream(pipeline_job.name)
            print(pipeline_job)
        except Exception as e:
            print(e)
            print('Failed with exception')
        self._verify_and_get_output_records(
            pipeline_job
        )
        assert_logged_params(
            pipeline_job.name,
            exp_name,
            prediction_dataset=[prediction_dataset],
            prediction_column_name=prediction_column_name,
            ground_truth_dataset=[ground_truth_dataset],
            ground_truth_column_name=ground_truth_column_name,
            separator=separator,
            regex_expr=regex_expr,
            strip_prefix=strip_prefix,
            strip_suffix=strip_suffix,
            template=template,
            script_path=script_path
        )

    def _get_pipeline_job(
        self,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        strip_prefix: str,
        strip_suffix: str,
        template: str,
        script_path: str,
        display_name: str,
        pipeline_file: str
    ) -> Job:
        """Get the pipeline job."""
        pipeline_job = load_yaml_pipeline(pipeline_file)
        # set the pipeline inputs
        pipeline_job.inputs.prediction_dataset = Input(type=AssetTypes.URI_FILE, path=prediction_dataset)
        pipeline_job.inputs.prediction_column_name = prediction_column_name
        if ground_truth_dataset:
            pipeline_job.inputs.ground_truth_dataset = Input(type=AssetTypes.URI_FILE, path=ground_truth_dataset)
        else:
            pipeline_job.inputs.ground_truth_dataset = None
        pipeline_job.inputs.ground_truth_column_name = ground_truth_column_name if ground_truth_column_name else None
        pipeline_job.inputs.separator = separator if separator else None
        pipeline_job.inputs.regex_expr = regex_expr if regex_expr else None
        pipeline_job.inputs.strip_prefix = strip_prefix if strip_prefix else None
        pipeline_job.inputs.strip_suffix = strip_suffix if strip_suffix else None
        pipeline_job.inputs.template = template if template else None
        if script_path:
            pipeline_job.inputs.script_path = Input(type=AssetTypes.URI_FILE, path=script_path)
        else:
            pipeline_job.inputs.script_path = None
        pipeline_job.display_name = display_name
        pipeline_job.description = "Post processor component testing"
        return pipeline_job

    def _verify_and_get_output_records(
        self,
        job: Job,
        output_dir: str = None
    ) -> None:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param output_dir: The local output directory to download pipeline outputs
        :type output_dir: str
        :return: None
        """
        if not output_dir:
            output_dir = os.getcwd()
        output_files = list(job.outputs.keys())
        for op_file in output_files:
            # output_name = job.outputs.output_dataset.port_name
            output_name = (job.outputs.get(op_file)).port_name
            download_outputs(
                job_name=job.name, output_name=output_name, download_path=output_dir
            )
