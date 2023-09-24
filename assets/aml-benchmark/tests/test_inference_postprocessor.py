# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Test script for Inference Postprocessor Component."""

import pytest
import json
import os
import glob
from typing import List, Dict, Any

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

from test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    assert_logged_params,
    get_src_dir,
    run_command
)

INPUT_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _verify_and_get_output_records(
    inputs: List[str],
    outputs: str
) -> List[Dict[str, Any]]:
    """Verify the output and get output records.

    :param inputs: The list of input file with absolute path.
    :param outputs: Either path to output file or output directory containing output files.
    :return: list of json records
    """
    if not os.path.isfile(outputs):
        output_files = glob.glob(outputs + '/**/*.jsonl', recursive=True)
    else:
        output_files = [outputs]
    assert len(output_files) == 1
    with open(output_files[0], "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)
    for input in inputs:
        with open(input, "r") as f:
            input_records = [json.loads(line) for line in f]
            input_row_count = len(input_records)
            assert input_row_count == output_row_count
    return output_records


class TestInferencePostprocessorComponent:
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
            self.test_inference_postprocessor_as_component.__name__,
            pipeline_file="inference_postprocessor_pipeline.yaml"
        )
        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(pipeline_job, experiment_name=self.EXP_NAME)
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)
        self._verify_and_get_output_records(
            pipeline_job, [prediction_dataset, ground_truth_dataset],
            output_dir=INPUT_PATH
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
        input_files: List[str],
        output_dir: str = None
    ) -> List[Dict[str, Any]]:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param output_dir: The local output directory to download pipeline outputs
        :type output_dir: str
        :return: output records
        :rtype: List[Dict[str, Any]]
        """
        output_name = job.outputs.output_dataset_result.port_name
        if not output_dir:
            output_dir = Constants.OUTPUT_DIR.format(os.getcwd(), output_name=output_name)
        else:
            output_dir = Constants.OUTPUT_DIR.format(output_dir=output_dir, output_name=output_name)
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        records = _verify_and_get_output_records(input_files, output_dir)
        return records


class TestInferencePostprocessorScript:
    """Testing the script."""

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
    def test_inference_postprocessor_as_script(
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
        output_dataset: str = os.path.join(INPUT_PATH, "postprocessed_output.jsonl"),
    ) -> None:
        """Inference Postprocessor script test."""
        argss = [
            "--prediction_dataset",
            prediction_dataset,
            "--prediction_column_name",
            prediction_column_name,
            "--output_dataset",
            output_dataset,
        ]
        if ground_truth_dataset is not None:
            argss.extend(["--ground_truth_dataset", ground_truth_dataset])
        if ground_truth_column_name is not None:
            argss.extend(["--ground_truth_column_name", ground_truth_column_name])
        if template is not None:
            argss.extend(["--template", f"'{template}'"])
        if script_path is not None:
            argss.extend(["--script_path", script_path])
        if separator is not None:
            argss.extend(["--separator", f"'{separator}'"])
        if regex_expr is not None:
            argss.extend(["--regex_expr", f"'{regex_expr}'"])
        if strip_prefix is not None:
            argss.extend(["--strip_prefix", f"'{strip_prefix}'"])
        elif strip_suffix is not None:
            argss.extend(["--strip_suffix", f"'{strip_suffix}'"])
        argss = " ".join(argss)
        src_dir = get_src_dir()
        cmd = f"cd {src_dir} && python -m inference_postprocessor.main {argss}"
        run_command(f"{cmd}")
        _verify_and_get_output_records([prediction_dataset, ground_truth_dataset], output_dataset)
        return

    def _verify_and_get_output_records(
        self,
        input_files: List[str],
        output_dataset: str
    ) -> List[Dict[str, Any]]:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param input_files: The list of input file with absolute path.
        :param output_dataset: path to output file.
        :rtype: List[Dict[str, Any]]
        """
        records = _verify_and_get_output_records(input_files, output_dataset)
        return records
