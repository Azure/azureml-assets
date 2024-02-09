# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Prompt Crafter Component."""

from typing import Optional
import os
import json
import subprocess
import pytest

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from azure.ai.ml.constants import AssetTypes

from .test_utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    get_src_dir,
    assert_exception_mssg,
    run_command,
    assert_logged_params,
)


def _verify_and_get_output_records(
    input_file: str,
    expected_output_file: str,
    is_ground_truth_col: bool = False,
) -> None:
    """Verify the output and get output records.

    :param input_file: The path to jsonl file containing input records.
    :param expected_output_file: file with expected outputs.
    """
    with open(input_file, "r") as f:
        input_records = [json.loads(line) for line in f]
        input_row_count = len(input_records)

    # if file exist
    assert os.path.isfile(expected_output_file) is True

    with open(expected_output_file, "r") as f:
        expected_output_records = [json.loads(line) for line in f]
        expected_output_row_count = len(expected_output_records)
        if is_ground_truth_col:
            for line in f:
                assert 'ground_truth' in json.loads(line).keys()
    assert input_row_count == expected_output_row_count
    return


# test patterns
_prompt_pattern_test = "Question:{{question}} \nChoices:(1) {{choices.text[0]}}\n(2) {{choices.text[1]}}\n(3) {{choices.text[2]}}\n(4) {{choices.text[3]}}\nThe answer is: "  # noqa: E501
_few_shot_pattern_test = "Question:{{question}} \nChoices:(1) {{choices.text[0]}}\n(2) {{choices.text[1]}}\n(3) {{choices.text[2]}}\n(4) {{choices.text[3]}}\nThe answer is: {{answerKey}}"  # noqa: E501
_output_pattern_test = "{{answerKey}}"
_ground_truth_column = "answerKey"
_additional_columns = "question,answerKey"


class TestPromptCrafterComponent:
    """Tests for prompt crafter component."""

    EXP_NAME = "prompt-crafter-test"

    @pytest.mark.parametrize(
        "test_data, prompt_type, n_shots, \
        few_shot_data, prompt_pattern, output_pattern, few_shot_pattern",
        [
            (
                Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 1,
                Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE,
                _prompt_pattern_test,
                _output_pattern_test,
                None,
            ),
            (
                Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 1,
                Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE,
                _prompt_pattern_test,
                _output_pattern_test,
                _few_shot_pattern_test,
            ),
            (
                Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "chat", 1,
                Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE,
                _prompt_pattern_test,
                _output_pattern_test,
                None,
            ),
            (
                Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "chat", 1,
                Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE,
                _prompt_pattern_test,
                _output_pattern_test,
                _few_shot_pattern_test,
            ),
        ]
    )
    def test_prompt_crafter_component(
        self,
        temp_dir: str,
        test_data: str,
        prompt_type: str,
        n_shots: int,
        few_shot_data: str,
        prompt_pattern: str,
        output_pattern: str,
        few_shot_pattern: Optional[str],
    ) -> None:
        """Prompt Crafter component test."""
        ml_client = get_mlclient()

        pipeline_job = self._get_pipeline_job(
            test_data,
            prompt_type,
            n_shots,
            few_shot_data,
            prompt_pattern,
            output_pattern,
            self.test_prompt_crafter_component.__name__,
            few_shot_pattern,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        self._verify_output(
            pipeline_job,
            test_data,
            output_dir=temp_dir
        )

        assert_logged_params(
            pipeline_job.name,
            self.EXP_NAME,
            prompt_type=prompt_type,
            n_shots=n_shots,
            prompt_pattern=prompt_pattern,
            output_pattern=output_pattern,
            few_shot_pattern=few_shot_pattern
        )

    def _get_pipeline_job(
        self,
        test_data: str,
        prompt_type: str,
        n_shots: int,
        few_shot_data: str,
        prompt_pattern: str,
        output_pattern: str,
        display_name: str,
        few_shot_pattern: Optional[str],
    ) -> Job:
        """Get the pipeline job.

        :return: The pipeline job.
        """
        if few_shot_pattern:
            pipeline_job = load_yaml_pipeline("prompt_crafter_pipeline_with_few_shot.yaml")
            pipeline_job.inputs.few_shot_pattern = few_shot_pattern
        else:
            pipeline_job = load_yaml_pipeline("prompt_crafter_pipeline.yaml")
        # set the pipeline inputs
        pipeline_job.inputs.test_data = Input(type=AssetTypes.URI_FILE, path=test_data)
        pipeline_job.inputs.prompt_type = prompt_type
        pipeline_job.inputs.prompt_pattern = prompt_pattern
        pipeline_job.inputs.few_shot_data = Input(type=AssetTypes.URI_FILE, path=few_shot_data)
        pipeline_job.inputs.n_shots = n_shots
        pipeline_job.inputs.output_pattern = output_pattern
        pipeline_job.display_name = display_name

        return pipeline_job

    def _verify_output(
        self,
        job: Job,
        input_file: str,
        output_dir: str
    ) -> None:
        """Verify the output and get output records.

        :param job: The pipeline job object.
        :type job: Job
        :param input_file: The path to input jsonl file
        :type input_file: str
        :param expected_output_file: The path to outut josnl file
        :type expected_output_file
        :param output_dir: Local output directory to download pipeline outputs.
        :type output_dir: str
        """
        output_name = job.outputs.output_file.port_name
        download_outputs(
            job_name=job.name, output_name=output_name,
            download_path=output_dir
        )
        output_file_path = Constants.OUTPUT_FILE_PATH.format(
            output_dir=output_dir,
            output_name=output_name,
            output_file_name=f"{output_name}.jsonl",  # taken from the pipeline's output path
        )
        _verify_and_get_output_records(
            input_file, output_file_path
        )
        return


class TestPromptCrafterScript:
    """Tests for prompt crafter script."""

    @pytest.mark.parametrize(
        "test_data, prompt_type, n_shots, \
        few_shot_data, prompt_pattern, few_shot_pattern, output_pattern, ground_truth_column, additional_columns",
        [
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 1,
             Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE, _prompt_pattern_test, _few_shot_pattern_test,
                _output_pattern_test, _ground_truth_column, _additional_columns),
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "chat", 1,
             Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE, _prompt_pattern_test, _few_shot_pattern_test,
                _output_pattern_test, _ground_truth_column, _additional_columns),
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 0,
             Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE, _prompt_pattern_test, None,
                _output_pattern_test, _ground_truth_column, _additional_columns),
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "chat", 0,
             Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE, _prompt_pattern_test, None,
                _output_pattern_test, _ground_truth_column, _additional_columns),
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 0,
                None, _prompt_pattern_test, None, _output_pattern_test, _ground_truth_column, _additional_columns),
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "chat", 0,
                None, _prompt_pattern_test, None, _output_pattern_test, _ground_truth_column, _additional_columns),
        ]
    )
    def test_valid_prompt_crafter(
        self,
        test_data: str,
        prompt_type: str,
        n_shots: int,
        few_shot_data: str,
        prompt_pattern: str,
        few_shot_pattern: Optional[str],
        output_pattern: str,
        ground_truth_column: str,
        additional_columns: str,
        output_file: str = os.path.join(
            os.path.dirname(__file__), 'data/prompt_crafter_output.jsonl'),
    ):
        """Test for valid input dataset."""
        # Run the script
        self._run_prompt_crafter_script(
            test_data=test_data,
            prompt_type=prompt_type,
            n_shots=n_shots,
            prompt_pattern=prompt_pattern,
            output_file=output_file,
            output_pattern=output_pattern,
            few_shot_data=few_shot_data,
            ground_truth_column=ground_truth_column,
            additional_columns=additional_columns,
            few_shot_pattern=few_shot_pattern,
        )

        # Verify the output file(s)
        _verify_and_get_output_records(
            test_data,
            output_file,
            is_ground_truth_col=True,
        )

    _error_mssg_few_shot_data_shortage = "Unable to find 10 few shots after 100 retries"
    _error_mssg_ground_truth_column_not_found = "Ground truth column is not present in the data"

    @pytest.mark.parametrize(
        "test_data, prompt_type, n_shots, \
        few_shot_data, prompt_pattern, output_pattern, \
        ground_truth_column, additional_columns, expected_error_message",
        [
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 10,
             Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE, _prompt_pattern_test,
             _output_pattern_test, _ground_truth_column, _additional_columns,
             _error_mssg_few_shot_data_shortage),
            (Constants.PROMPTCRAFTER_SAMPLE_INPUT_FILE, "completions", 1,
             Constants.PROMPTCRAFTER_SAMPLE_FEWSHOT_FILE, _prompt_pattern_test,
             _output_pattern_test, "invalid_column", None, _error_mssg_ground_truth_column_not_found),
        ]
    )
    def test_invalid_prompt_crafter(
        self,
        test_data: str,
        prompt_type: str,
        n_shots: int,
        few_shot_data: str,
        prompt_pattern: str,
        output_pattern: str,
        ground_truth_column: str,
        additional_columns: str,
        expected_error_message: str,
        output_file: str = os.path.join(
            os.path.dirname(__file__), 'data/prompt_crafter_output.jsonl'),
    ):
        """Test for valid input dataset."""
        try:
            # Run the script
            self._run_prompt_crafter_script(
                test_data=test_data,
                prompt_type=prompt_type,
                n_shots=n_shots,
                prompt_pattern=prompt_pattern,
                output_file=output_file,
                output_pattern=output_pattern,
                few_shot_data=few_shot_data,
                ground_truth_column=ground_truth_column,
                additional_columns=additional_columns,
            )
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, expected_error_message)

    def _run_prompt_crafter_script(
        self,
        test_data: str,
        prompt_type: str,
        n_shots: int,
        prompt_pattern: str,
        output_file: str,
        ground_truth_column: Optional[str] = None,
        additional_columns: Optional[str] = None,
        few_shot_separator: Optional[str] = None,
        prefix: Optional[str] = None,
        system_message: Optional[str] = None,
        few_shot_data: Optional[str] = None,
        random_seed: Optional[int] = 0,
        output_pattern: str = "{{target}}",
        few_shot_pattern: Optional[str] = None,
    ) -> None:
        """
        Run the prompt crafter script.

        param: test_data: Path to jsonl would be used to generate prompts.
        param: prompt_type: Type of prompt to generate.
        param: n_shots: Number of shots to use for few-shot prompts.
        param: output_pattern: Pattern to use for output prompts.
        param: prompt_pattern: Pattern to use for prompts.
        param: output_data: Path to jsonl with generated prompts.
        param: few_shot_separator: Separator to use for few-shot prompts.
        param: prefix: Prefix to use for prompts.
        param: system_message: System message to use for prompts.
        param: few_shot_data: Path to jsonl to generate n-shot prompts.
        param: random_seed: Random seed to use for prompts.
        param: few_shot_pattern: Pattern to use for few shot prompts
        """
        src_dir = get_src_dir()
        args = [
            f"cd {src_dir} &&",
            "python -m aml_benchmark.prompt_crafter.main",
            "--system_message",
            f"{system_message}",
            "--random_seed",
            f"{random_seed}",
            "--output_file",
            f"{output_file}",
            "--few_shot_separator",
            f"{few_shot_separator}",
            "--prefix",
            f"{prefix}",
        ]
        if test_data is not None:
            args.extend(["--test_data", f"{test_data}"])
        if prompt_type is not None:
            args.extend(["--prompt_type", f"{prompt_type}"])
        if n_shots is not None:
            args.extend(["--n_shots", f"{n_shots}"])
        if prompt_pattern is not None:
            args.extend(["--prompt_pattern", f'"{prompt_pattern}"'])
        if few_shot_pattern is not None:
            args.extend(["--few_shot_pattern", f'"{few_shot_pattern}"'])
        if output_pattern is not None:
            args.extend(["--output_pattern", f'"{output_pattern}"'])
        if few_shot_data is not None:
            args.extend(["--few_shot_data", f'"{few_shot_data}"'])
        if ground_truth_column is not None:
            args.extend(["--ground_truth_column_name", f"{ground_truth_column}"])
        if additional_columns is not None:
            args.extend(["--additional_columns", f"{additional_columns}"])

        run_command(str(" ".join(args)))
