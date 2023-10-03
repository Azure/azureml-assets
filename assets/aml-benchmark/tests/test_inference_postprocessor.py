# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# ---------------------------------------------------------

"""Test script for Inference Postprocessor Component."""

import pytest
import json
import os
import glob
import subprocess

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
    run_command,
    assert_exception_mssg
)


INPUT_PATH = os.path.join(os.path.dirname(__file__), 'data')


def _verify_and_get_output_records(
    dataset_name: str,
    expected_output_file: str,
    outputs: str,
    input_file: str = None
) -> None:
    """Verify the output and get output records.

    :param dataset_name: The path to jsonl file passed as input_dataset in pipeline.
    :param input_file: The path to jsonl file passed as input_dataset in pipeline or scripts.
    :param expected_output_file: The path to jsonl file containing expected outputs.
    :param outputs: Either path to output file or output directory containing output files.
    """
    if input_file:
        with open(input_file, "r") as f:
            input_records = [json.loads(line) for line in f]
            input_row_count = len(input_records)
    expected_output_records = []
    with open(expected_output_file, "r") as f:
        for line in f:
            out = json.loads(line)
            if out.get('name') == dataset_name:
                del out['name']
                expected_output_records.append(out)
    expected_output_row_count = len(expected_output_records)
    if not os.path.isfile(outputs):
        output_files = glob.glob(outputs + '/**/*.jsonl', recursive=True)
    else:
        output_files = [outputs]
    assert len(output_files) == 1
    with open(output_files[0], "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)
    if input_file:
        assert input_row_count == output_row_count == expected_output_row_count
    else:
        assert output_row_count == expected_output_row_count
    assert output_records == expected_output_records
    return


class TestInferencePostprocessorComponent:
    """Test Inference Postprocessor."""

    EXP_NAME = "postprocessor-test"

    @pytest.mark.parametrize(
        "dataset_name, prediction_dataset, prediction_column_name, ground_truth_dataset, ground_truth_column_name, \
        separator, regex_expr, extract_value_at_index, strip_prefix, strip_suffix, template, script_path, \
        pred_probs_dataset, encoder_config",
        [
            (
                "gsm8k",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", None, None,
                None, None, None, """{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}""",
                None, None, None
            ),
            (
                "human-eval",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "samples",
                None, None, None, '^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)',
                0, None, None, None, None, None, None
            ),
        ],
    )
    def test_inference_postprocessor_as_component(
        self,
        dataset_name: str,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        extract_value_at_index: int,
        strip_prefix: str,
        strip_suffix: str,
        template: str,
        script_path: str,
        pred_probs_dataset: str,
        encoder_config: str,
    ) -> None:
        """Inference Postprocessor component test."""
        if ground_truth_dataset:
            with open(
                os.path.join(
                    os.path.dirname(
                        Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"), "w"
                    ) as writer:
                with open(Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "r") as reader:
                    for line in reader:
                        out_row = json.loads(line)
                        if out_row.get('name') == dataset_name:
                            del out_row['name']
                            writer.write(json.dumps(out_row) + "\n")
            ground_truth_dataset = os.path.join(
                os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                "process_one_example.jsonl"
            )
        with open(
            os.path.join(
                os.path.dirname(
                    Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_prediction_example.jsonl"), "w"
                ) as writer:
            with open(Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        prediction_dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_prediction_example.jsonl"
        )
        ml_client = get_mlclient()
        exp_name = f"{self.EXP_NAME}"
        pipeline_job = self._get_pipeline_job(
            prediction_dataset,
            prediction_column_name,
            ground_truth_dataset,
            ground_truth_column_name,
            separator,
            regex_expr,
            extract_value_at_index,
            strip_prefix,
            strip_suffix,
            template,
            script_path,
            pred_probs_dataset,
            encoder_config,
            self.test_inference_postprocessor_as_component.__name__,
            pipeline_file="inference_postprocessor_pipeline.yaml"
        )
        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(pipeline_job, experiment_name=self.EXP_NAME)
        ml_client.jobs.stream(pipeline_job.name)
        self._verify_and_get_output_records(
            pipeline_job, dataset_name, ground_truth_dataset,
            Constants.POSTPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE,
            output_dir=os.path.join(os.path.dirname(__file__), 'data')
        )
        assert_logged_params(
            pipeline_job.name,
            exp_name,
            prediction_dataset=[prediction_dataset],
            prediction_column_name=prediction_column_name,
            ground_truth_dataset=[ground_truth_dataset] if ground_truth_dataset else None,
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
        extract_value_at_index: int,
        strip_prefix: str,
        strip_suffix: str,
        template: str,
        script_path: str,
        pred_probs_dataset: str,
        encoder_config: str,
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
        pipeline_job.inputs.extract_value_at_index = extract_value_at_index if extract_value_at_index else None
        pipeline_job.inputs.strip_prefix = strip_prefix if strip_prefix else None
        pipeline_job.inputs.strip_suffix = strip_suffix if strip_suffix else None
        pipeline_job.inputs.template = template if template else None
        if script_path:
            pipeline_job.inputs.script_path = Input(type=AssetTypes.URI_FILE, path=script_path)
        else:
            pipeline_job.inputs.script_path = None
        if pred_probs_dataset:
            pipeline_job.inputs.prediction_probabilities_dataset = Input(
                type=AssetTypes.URI_FILE, path=pred_probs_dataset
            )
        else:
            pipeline_job.inputs.prediction_probabilities_dataset = None
        pipeline_job.inputs.encoder_config = encoder_config if encoder_config else None
        pipeline_job.display_name = display_name
        pipeline_job.description = "Post processor component testing"
        return pipeline_job

    def _verify_and_get_output_records(
        self,
        job: Job,
        dataset_name: str = None,
        input_file: str = None,
        expected_output_file: str = None,
        output_dir: str = None
    ) -> None:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param dataset_name: The path to jsonl file passed as input_dataset in pipeline.
        :type dataset_name: str
        :param input_file: The path to jsonl file passed as input_dataset in pipeline.
        :type input_file: str
        :param expected_output_file: The path to josnl file containing expected outputs for given inputs.
        :type expected_output_file
        :param output_dir: The local output directory to download pipeline outputs.
        :type output_dir: str
        """
        output_name = job.outputs.output_dataset_result.port_name
        if not output_dir:
            output_dir = Constants.OUTPUT_DIR.format(os.getcwd(), output_name=output_name)
        else:
            output_dir = Constants.OUTPUT_DIR.format(output_dir=output_dir, output_name=output_name)
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        _verify_and_get_output_records(
            dataset_name, expected_output_file, output_dir, input_file=input_file
        )
        return


class TestInferencePostprocessorScript:
    """Testing the script."""

    @pytest.mark.parametrize(
        "dataset_name, prediction_dataset, prediction_column_name, ground_truth_dataset, ground_truth_column_name, \
        separator, regex_expr, extract_value_at_index, strip_prefix, strip_suffix, template, script_path, \
        pred_probs_dataset, encoder_config",
        [
            (
                "gsm8k",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "prediction",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "final_answer", None, None,
                None, None, None, """{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}""",
                None, None, None
            ),
            (
                "human-eval",
                Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "samples",
                None, None, None, '^(.*?)(\nclass|\ndef|\n#|\nif|\nprint|$)',
                0, None, None, None, None, None, None
            ),
        ],
    )
    def test_inference_postprocessor_as_script(
        self,
        dataset_name: str,
        prediction_dataset: str,
        prediction_column_name: str,
        ground_truth_dataset: str,
        ground_truth_column_name: str,
        separator: str,
        regex_expr: str,
        extract_value_at_index: int,
        strip_prefix: str,
        strip_suffix: str,
        template: str,
        script_path: str,
        pred_probs_dataset: str,
        encoder_config: str,
        output_dataset: str = os.path.join(INPUT_PATH, "postprocessed_output.jsonl"),
    ) -> None:
        """Inference Postprocessor script test."""
        if ground_truth_dataset:
            with open(
                os.path.join(
                    os.path.dirname(
                        Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_example.jsonl"), "w"
                    ) as writer:
                with open(Constants.POSTPROCESS_SAMPLE_EXAMPLES_GROUND_TRUTH_FILE, "r") as reader:
                    for line in reader:
                        out_row = json.loads(line)
                        if out_row.get('name') == dataset_name:
                            del out_row['name']
                            writer.write(json.dumps(out_row) + "\n")
            ground_truth_dataset = os.path.join(
                os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
                "process_one_example.jsonl"
            )
        with open(
            os.path.join(
                os.path.dirname(
                    Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE), "process_one_prediction_example.jsonl"), "w"
                ) as writer:
            with open(Constants.POSTPROCESS_SAMPLE_EXAMPLES_INFERENCE_FILE, "r") as reader:
                for line in reader:
                    out_row = json.loads(line)
                    if out_row.get('name') == dataset_name:
                        del out_row['name']
                        writer.write(json.dumps(out_row) + "\n")
        prediction_dataset = os.path.join(
            os.path.dirname(Constants.PROCESS_SAMPLE_EXAMPLES_INPUT_FILE),
            "process_one_prediction_example.jsonl"
        )
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
        if extract_value_at_index is not None:
            argss.extend(["--extract_value_at_index", str(extract_value_at_index)])
        if strip_prefix is not None:
            argss.extend(["--strip_prefix", f"'{strip_prefix}'"])
        if strip_suffix is not None:
            argss.extend(["--strip_suffix", f"'{strip_suffix}'"])
        if pred_probs_dataset is not None:
            argss.extend(["--strip_suffix", f"'{strip_suffix}'"])
        if encoder_config is not None:
            argss.extend(["--strip_suffix", f"'{strip_suffix}'"])
        argss = " ".join(argss)
        src_dir = get_src_dir()
        cmd = f"cd {src_dir} && python -m inference_postprocessor.main {argss}"
        run_command(f"{cmd}")
        _verify_and_get_output_records(
            dataset_name, Constants.POSTPROCESS_SAMPLE_EXAMPLES_EXPECTED_OUTPUT_FILE,
            output_dataset, input_file=ground_truth_dataset
        )
        return

    @pytest.mark.parametrize(
        "prediction_dataset, prediction_column_name, template",
        [
            (
                os.path.join(INPUT_PATH, "sample_predictions.jsonl"), "prediction",
                '{{prediction.split("\n\n")[0].split(" ")[-1].rstrip(".")}}'
            ),
        ],
    )
    def test_invalid_inputs(
        self,
        prediction_dataset: str,
        prediction_column_name: str,
        template: str
    ):
        """Test the exceptions raise during inputs validation."""
        invalid_prediction_dataset_error_mssg = (
            "the following arguments are required: --prediction_dataset"
        )
        invalid_jsonl_dataset_mssg = (
            "No .jsonl files found in the given prediction dataset."
        )
        invalid_prediction_colname_error_mssg = (
            "the following arguments are required: --prediction_column_name"
        )
        invalid_user_script_mssg = (
            "Please provide python script containing your custom postprocessor logic."
        )
        src_dir = get_src_dir()
        try:
            argss = " ".join([
                "--prediction_column_name", prediction_column_name,
                "--template", f"'{template}'"
            ])
            cmd = f"cd {src_dir} && python -m inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            out_message = e.output.strip()
            assert invalid_prediction_dataset_error_mssg in out_message
        try:
            argss = " ".join([
                "--prediction_dataset", prediction_dataset,
                "--template", f"'{template}'"
            ])
            cmd = f"cd {src_dir} && python -m inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            out_message = e.output.strip()
            assert invalid_prediction_colname_error_mssg in out_message

        dummy_dataset_path = os.path.join(os.getcwd(), "prediction_dataset_path")
        os.system(f"mkdir {dummy_dataset_path}")
        try:
            argss = " ".join([
                "--prediction_dataset", dummy_dataset_path,
                "--prediction_column_name", prediction_column_name,
                "--template", f"'{template}'"
            ])
            cmd = f"cd {src_dir} && python -m inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_jsonl_dataset_mssg)

        dummy_script_path = os.path.join(os.getcwd(), "user_script.json")
        os.system(f"touch {dummy_script_path}")
        try:
            argss = " ".join([
                "--prediction_dataset", prediction_dataset,
                "--prediction_column_name", prediction_column_name,
                "--script_path", dummy_script_path
            ])
            cmd = f"cd {src_dir} && python -m inference_postprocessor.main {argss}"
            run_command(f"{cmd}")
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, invalid_user_script_mssg)
