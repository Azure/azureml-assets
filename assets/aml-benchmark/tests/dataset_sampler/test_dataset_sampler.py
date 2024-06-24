# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Dataset Sampler Component."""

from typing import Union, List, Dict, Any, Optional
import json
import os
import shutil
import uuid
import random

import mltable
import pytest
from azure.ai.ml.entities import Job
from azure.ai.ml import Input

from ..test_utils import (
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
    input_paths: List[str],
    output_dir: str,
    sampling_style: str = "head",
    sampling_ratio: Union[float, None] = None,
    n_samples: Union[int, None] = None,
) -> List[Dict[str, Any]]:
    """Verify the output and get output records.

    :param input_paths: The list of input file paths.
    :param output_dir: The output directory.
    :param sampling_style: The sampling style, defaults to "head".
    :param sampling_ratio: The sampling ratio, defaults to None.
    :param n_samples: The number of samples, defaults to None.
    :return: list of json records
    """
    output_paths = [
        os.path.join(output_dir, file)
        for file in os.listdir(output_dir)
        if file.endswith(".jsonl")
    ]
    assert len(output_paths) == 1

    # Read the output file
    with open(output_paths[0], "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)
    expected_row_count = 0
    expected_rows = []

    for input_path in input_paths:
        # Read the input file
        with open(input_path, "r") as f:
            input_records = [json.loads(line) for line in f]
        input_row_count = len(input_records)

        # Check row count and records
        if n_samples:
            curr_exp_row_count = n_samples
        else:
            curr_exp_row_count = int(input_row_count * sampling_ratio)
        expected_row_count += curr_exp_row_count

        if sampling_style == "head":
            expected_rows += input_records[:curr_exp_row_count]
        elif sampling_style == "tail":
            expected_rows += input_records[-curr_exp_row_count:]
        elif sampling_style == "duplicate":
            expected_rows += input_records + input_records[:curr_exp_row_count - input_row_count]

    assert output_row_count == expected_row_count
    if sampling_style != "random":
        assert output_records == expected_rows
    return output_records


class TestDatasetSamplerComponent:
    """Tests for dataset sampler component."""

    EXP_NAME = "dataset-sampler-test"

    @pytest.mark.parametrize(
        "sampling_style, sampling_ratio, n_samples",
        [
            ("head", 0.1, None),
            ("tail", None, 6),
            ("random", None, 5),
            ("duplicate", 1.2, None),
        ],
    )
    def test_dataset_sampler_component(
        self,
        temp_dir: str,
        sampling_style: str,
        sampling_ratio: Union[float, None],
        n_samples: Union[int, None],
    ) -> None:
        """Dataset Sampler component test."""
        ml_client = get_mlclient()

        input_file_path = self._create_input_file(temp_dir)
        pipeline_job = self._get_pipeline_job(
            input_file_path,
            sampling_style,
            sampling_ratio,
            n_samples,
            self.test_dataset_sampler_component.__name__,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        self._verify_and_get_output_records(
            pipeline_job,
            sampling_style,
            sampling_ratio,
            n_samples,
            [input_file_path],
            temp_dir,
        )
        assert_logged_params(
            pipeline_job.name,
            self.EXP_NAME,
            dataset=[input_file_path],
            sampling_style=sampling_style,
            sampling_ratio=sampling_ratio,
            n_samples=n_samples,
            random_seed=0 if sampling_style == "random" else None,
        )

    @pytest.mark.parametrize(
        "sampling_ratio, n_samples",
        [
            (0.1, None),
            (None, 8),
        ],
    )
    def test_reproducibility_for_random(
        self,
        temp_dir: str,
        sampling_ratio: Union[float, None],
        n_samples: Union[int, None],
    ) -> None:
        """Test reproducibility when `sampling_style` is `random`."""
        sampling_style = "random"
        ml_client = get_mlclient()
        output_records = []
        input_file_path = self._create_input_file(temp_dir)

        # run the pipeline twice to verify reproducibility
        for _ in range(2):
            pipeline_job = self._get_pipeline_job(
                input_file_path,
                sampling_style,
                sampling_ratio,
                n_samples,
                self.test_reproducibility_for_random.__name__,
            )
            pipeline_job.settings.force_rerun = True
            # submit the pipeline job
            pipeline_job = ml_client.create_or_update(
                pipeline_job, experiment_name=self.EXP_NAME
            )
            ml_client.jobs.stream(pipeline_job.name)
            print(pipeline_job)

            output_records.append(
                self._verify_and_get_output_records(
                    pipeline_job,
                    sampling_style,
                    sampling_ratio,
                    n_samples,
                    [input_file_path],
                    temp_dir,
                )
            )
            assert_logged_params(
                pipeline_job.name,
                self.EXP_NAME,
                dataset=[input_file_path],
                sampling_style=sampling_style,
                sampling_ratio=sampling_ratio,
                n_samples=n_samples,
                random_seed=0 if sampling_style == "random" else None,
            )

        # verify the output of two runs are the same
        assert output_records[0] == output_records[1]

    def _create_input_file(self, directory: str) -> str:
        """Create .jsonl input file.

        :param directory: The existing directory to create the file in.
        :return: The created input file path.
        """
        file_name = uuid.uuid4().hex + ".jsonl"
        file_path = os.path.join(directory, file_name)

        # create input file with 20 records
        with open(file_path, "w") as f:
            for i in range(20):
                record = {
                    "id": i,
                    "name": f"Person {uuid.uuid4().hex}",
                    "age": random.randint(18, 50),
                }
                f.write(json.dumps(record) + "\n")

        return file_path

    def _get_pipeline_job(
        self,
        input_file_path: str,
        sampling_style: str,
        sampling_ratio: Union[float, None],
        n_samples: Union[int, None],
        display_name: str,
    ) -> Job:
        """Get the pipeline job.

        :param input_file_path: The input file path.
        :param sampling_style: The sampling style.
        :param sampling_ratio: The sampling ratio.
        :param n_samples: The number of samples.
        :param display_name: The display name for job.
        :return: The pipeline job.
        """
        pipeline_job = load_yaml_pipeline("dataset_sampler_pipeline.yaml")

        # set the pipeline inputs
        pipeline_job.inputs.dataset = Input(type="uri_file", path=input_file_path)
        pipeline_job.inputs.sampling_style = sampling_style
        pipeline_job.inputs.sampling_ratio = sampling_ratio
        pipeline_job.inputs.n_samples = n_samples
        pipeline_job.display_name = display_name

        return pipeline_job

    def _verify_and_get_output_records(
        self,
        job: Job,
        sampling_style: str,
        sampling_ratio: Union[float, None],
        n_samples: Union[int, None],
        input_file_paths: List[str],
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        """Verify the output and get output records.

        :param job: The job object.
        :param sampling_style: The sampling style.
        :param sampling_ratio: The sampling ratio.
        :param n_samples: The number of samples.
        :param input_file_paths: The list of input file paths.
        :param output_dir: The existing output directory to download the output in.
        :return: output records
        """
        output_name = job.outputs.output_dataset.port_name
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        output_dir = Constants.OUTPUT_DIR.format(
            output_dir=output_dir,
            output_name=output_name,
        )

        return _verify_and_get_output_records(
            input_file_paths, output_dir, sampling_style, sampling_ratio, n_samples
        )


class TestDatasetSamplerScript:
    """Tests for dataset sampler script."""

    @pytest.mark.parametrize("dataset_type", ["uri_file", "uri_folder"])
    def test_invalid_dataset(self, temp_dir: str, dataset_type: str):
        """Test for invalid input dataset."""
        # Create input file and expected exception message
        if dataset_type == "uri_file":
            # Create an input file with invalid extension i.e. `.json` instead of `.jsonl`
            input_file_path = self._create_input_file(
                temp_dir, file_name="temp_test.json"
            )
            expected_exception_mssg = f"No .jsonl files found in {input_file_path}."
        else:
            # Create two `.json` files in the folder, folder as input is now invalid
            input_file_path = self._create_input_file(
                temp_dir, file_name="temp_test_1.json"
            )
            input_file_path = self._create_input_file(
                temp_dir, file_name="temp_test_2.json"
            )
            expected_exception_mssg = f"No .jsonl files found in {temp_dir}."
            input_file_path = temp_dir

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path)
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.skip(
        "Dataset Sampler does not care if the jsonl file is valid or not."
    )
    def test_invalid_jsonl_file_as_dataset(self, temp_dir: str):
        """Test for invalid jsonl file as dataset."""
        # Create input file and expected exception message
        input_file_path = os.path.join(temp_dir, "invalid_data.jsonl")
        lines = [
            '{"name": "Person A", "age": 25}',
            "Invalid JSON Line",
            '{"name": "Person B", "age": 35}',
        ]
        with open(input_file_path, "w") as file:
            for line in lines:
                file.write(line + "\n")

        expected_exception_mssg = (
            f"Input file '{input_file_path}' is not a valid .jsonl file."
        )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path)
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize(
        "dataset_type, sampling_style, n_samples, sampling_ratio",
        [
            ("uri_folder", "head", 5, None),
            ("mltable", "tail", None, 0.3),
            ("uri_folder", "duplicate", None, 1.3),
            ("uri_file", "random", 10, None),
        ]
    )
    def test_valid_dataset(
        self,
        temp_dir,
        dataset_type: str,
        sampling_style: str,
        n_samples: Optional[int],
        sampling_ratio: Optional[float]
    ):
        """Test for valid input dataset."""
        # Create input file
        shutil.copy(Constants.SAMPLER_INPUT_FILE_1, temp_dir)

        dataset = os.path.join(temp_dir, Constants.SAMPLER_INPUT_FILE_1.split("/")[-1])
        input_file_paths = [dataset]
        if dataset_type == "uri_folder":
            shutil.copy(Constants.SAMPLER_INPUT_FILE_2, temp_dir)
            input_file_paths.append(os.path.join(temp_dir, Constants.SAMPLER_INPUT_FILE_2.split("/")[-1]))
            dataset = temp_dir
        elif dataset_type == "mltable":
            paths = [
                {
                    "file": dataset,
                }
            ]
            mltable.from_json_lines_files(paths).save(temp_dir)
            dataset = temp_dir

        # Run the script
        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        file_path = os.path.join(out_dir, "out.jsonl")
        self._run_sampler_script(
            dataset,
            n_samples=n_samples,
            output_dataset=file_path,
            sampling_style=sampling_style,
            sampling_ratio=sampling_ratio
        )

        # Verify the output file(s)
        _verify_and_get_output_records(
            input_file_paths,
            out_dir,
            n_samples=n_samples,
            sampling_style=sampling_style,
            sampling_ratio=sampling_ratio
        )

    @pytest.mark.parametrize("sampling_ratio", [0, -0.6, 1.1, None])
    def test_invalid_sampling_ratio(
        self, temp_dir: str, sampling_ratio: Union[float, None]
    ):
        """Test for invalid sampling ratio."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        if sampling_ratio is None:
            expected_exception_mssg = (
                "Either 'sampling_ratio' or 'n_samples' must be specified."
            )
        elif sampling_ratio > 1:
            expected_exception_mssg = (
                "Invalid sampling_ratio: sampling_ratio > 1 only allowed for sampling style `duplicate`. "
                "Received: sampling_style=head."
            )
        else:
            expected_exception_mssg = (
                f"Invalid sampling_ratio: {float(sampling_ratio)}. Please specify a positive float number."
            )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path, sampling_ratio=sampling_ratio)
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize("n_samples", [0, -1, None])
    def test_invalid_n_samples(self, temp_dir: str, n_samples: Union[int, None]):
        """Test for invalid n_samples."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        if n_samples is None:
            expected_exception_mssg = (
                "Either 'sampling_ratio' or 'n_samples' must be specified."
            )
        else:
            expected_exception_mssg = (
                f"Invalid n_samples: {n_samples}. Please specify positive integer."
            )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path, n_samples=n_samples)
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    def test_sampling_ratio_n_samples_together(self, temp_dir: str):
        """Test for sampling ratio and n_samples together."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        expected_exception_mssg = (
            "Only one of 'sampling_ratio' or 'n_samples' can be specified."
        )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path, sampling_ratio=0.2, n_samples=10)
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize("sampling_style", ["haphazard"])
    def test_invalid_sampling_style(self, temp_dir: str, sampling_style: str):
        """Test for invalid sampling style."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        # the message below is the default message for invalid choice directly from argparse
        expected_exception_mssg = (
            f"argument --sampling_style: invalid choice: '{sampling_style}' "
            "(choose from 'head', 'tail', 'random', 'duplicate')"
        )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(
                input_file_path, sampling_style=sampling_style, sampling_ratio=0.2
            )
        except RuntimeError as e:
            exception_message = str(e)
            assert expected_exception_mssg in exception_message

    @pytest.mark.parametrize("sampling_style", ["head", "tail", "duplicate"])
    def test_random_seed_for_non_random_sampling(
        self, temp_dir: str, sampling_style: str
    ):
        """Test random seed for non random sampling."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        expected_warning_mssg = (
            f"Received random_seed: {0}, but it won't be used. It is used only when 'sampling_style' is 'random'."
        )

        out_dir = os.path.join(temp_dir, "out")
        os.makedirs(out_dir, exist_ok=True)
        out_file_path = os.path.join(out_dir, "out.jsonl")

        # Run the script and verify the exception
        try:
            self._run_sampler_script(
                input_file_path,
                sampling_style=sampling_style,
                sampling_ratio=0.2,
                output_dataset=out_file_path,
            )
        except RuntimeError as e:
            exception_message = str(e)
            assert expected_warning_mssg in exception_message

    def _create_input_file(self, directory: str, file_name: str) -> str:
        """Get the input file path.

        :param directory: The existing directory to create the file in.
        :type directory: str
        :param file_name: The file name with extension, either `.json` or `.jsonl`.
        :type file_name: str
        :return: The created input file path.
        :rtype: str
        """
        file_path = os.path.join(directory, file_name)
        file_content = json.dumps({"a": 1, "b": 2, "c": 3})
        with open(file_path, "w") as f:
            f.write(file_content)
        return file_path

    def _run_sampler_script(
        self,
        dataset: str,
        sampling_style: str = "head",
        sampling_ratio: Union[float, None] = None,
        n_samples: Union[int, None] = None,
        random_seed: int = 0,
        output_dataset: str = "out.jsonl",
    ) -> None:
        """
        Run the dataset sampler script with the given arguments.

        :param dataset: The input file or directory.
        :type dataset: str
        :param sampling_style: The sampling style, defaults to "head"
        :type sampling_style: str, optional
        :param sampling_ratio: The sampling ratio, defaults to None
        :type sampling_ratio: Union[float, None], optional
        :param n_samples: The number of samples, defaults to None
        :type n_samples: Union[int, None], optional
        :param random_seed: The random seed, defaults to 0
        :type random_seed: int, optional
        :param output_dataset: The output file path, defaults to "out.jsonl"
        :type output_dataset: str, optional
        :return: None
        :rtype: NoneType
        """
        src_dir = get_src_dir()
        args = [
            f"cd {src_dir} &&",
            "python -m aml_benchmark.dataset_sampler.main",
            "--dataset",
            dataset,
            "--sampling_style",
            sampling_style,
            "--random_seed",
            str(random_seed),
            "--output_dataset",
            output_dataset,
        ]
        if sampling_ratio is not None:
            args.extend(["--sampling_ratio", str(sampling_ratio)])
        if n_samples is not None:
            args.extend(["--n_samples", str(n_samples)])

        run_command(" ".join(args))
