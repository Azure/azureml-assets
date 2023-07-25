# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
from utils import load_yaml_pipeline, get_mlclient, Constants, download_outputs, get_mlflow_logged_params
from azure.ai.ml.entities import Job
from azure.ai.ml import Input
from typing import Union, List, Dict, Any
import pytest
import json
import os
import subprocess
import hashlib
import shutil
import mltable


def _verify_and_get_output_records(
    output_path: str,
    sampling_style: str = "head",
    sampling_ratio: Union[float, None] = None,
    n_samples: Union[int, None] = None,
) -> List[Dict[str, Any]]:
    """Verify the output and get output records.

    :param output_path: The output file path.
    :type output_path: str
    :param sampling_style: The sampling style, defaults to "head".
    :type sampling_style: str, optional
    :param sampling_ratio: The sampling ratio, defaults to None.
    :type sampling_ratio: Union[float, None], optional
    :param n_samples: The number of samples, defaults to None.
    :type n_samples: Union[int, None], optional
    :return: output records
    :rtype: List[Dict[str, Any]]
    """
    # Read the input file
    with open(Constants.INPUT_FILE_PATH, "r") as f:
        input_records = [json.loads(line) for line in f]

    # Read the output file
    with open(output_path, "r") as f:
        output_records = [json.loads(line) for line in f]
    output_row_count = len(output_records)

    # Check row count and records
    if n_samples:
        assert output_row_count == n_samples
    else:
        assert output_row_count == int(len(input_records) * sampling_ratio)

        if sampling_style == "head":
            assert output_records == input_records[:output_row_count]
        elif sampling_style == "tail":
            assert output_records == input_records[-output_row_count:]
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

        pipeline_job = self._get_pipeline_job(
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
            pipeline_job, sampling_style, sampling_ratio, n_samples, temp_dir
        )
        self._verify_logged_params(
            pipeline_job.name, sampling_style, sampling_ratio, n_samples
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

        # run the pipeline twice to verify reproducibility
        for _ in range(2):
            pipeline_job = self._get_pipeline_job(
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
                    temp_dir,
                )
            )
            self._verify_logged_params(
                pipeline_job.name, sampling_style, sampling_ratio, n_samples
            )

        # verify the output of two runs are the same
        assert output_records[0] == output_records[1]

    def _get_pipeline_job(
        self,
        sampling_style: str,
        sampling_ratio: Union[float, None],
        n_samples: Union[int, None],
        display_name: str,
    ) -> Job:
        """Get the pipeline job.

        :param sampling_style: The sampling style.
        :type sampling_style: str
        :param sampling_ratio: The sampling ratio.
        :type sampling_ratio: Union[float, None]
        :param n_samples: The number of samples.
        :type n_samples: Union[int, None]
        :param display_name: The display name for job.
        :type display_name: str
        :return: The pipeline job.
        :rtype: Job
        """
        pipeline_job = load_yaml_pipeline("dataset_sampler_pipeline.yaml")

        # set the pipeline inputs
        pipeline_job.inputs.dataset = Input(type="uri_file", path=Constants.INPUT_FILE_PATH)
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
        output_dir: str,
    ) -> List[Dict[str, Any]]:
        """Verify the output and get output records.

        :param job: The job object.
        :type job: Job
        :param sampling_style: The sampling style.
        :type sampling_style: str
        :param sampling_ratio: The sampling ratio.
        :type sampling_ratio: Union[float, None]
        :param n_samples: The number of samples.
        :type n_samples: Union[int, None]
        :param output_dir: The existing output directory to download the output in.
        :type output_dir: str
        :return: output records
        :rtype: List[Dict[str, Any]]
        """
        output_name = job.outputs.output_dataset.port_name
        download_outputs(
            job_name=job.name, output_name=output_name, download_path=output_dir
        )
        output_file_path = Constants.OUTPUT_FILE_PATH.format(
            output_dir=output_dir,
            output_name=output_name,
            output_file_name="output_dataset.jsonl"  # taken from the pipeline's output path
        )
        return _verify_and_get_output_records(
            output_file_path, sampling_style, sampling_ratio, n_samples
        )

    def _verify_logged_params(
        self,
        job_name: str,
        sampling_style: str = "head",
        sampling_ratio: Union[float, None] = None,
        n_samples: Union[int, None] = None,
        random_seed: int = 0,
    ) -> None:
        """Verify the logged parameters.

        :param job_name: The job name.
        :type job_name: str
        :param sampling_style: The sampling style, defaults to "head"
        :type sampling_style: str, optional
        :param sampling_ratio: The sampling ratio, defaults to None
        :type sampling_ratio: Union[float, None], optional
        :param n_samples: The number of samples, defaults to None
        :type n_samples: Union[int, None], optional
        :param random_seed: The random seed, defaults to 0
        :type random_seed: int, optional
        :return: None.
        :rtype: NoneType
        """
        logged_params = get_mlflow_logged_params(job_name, self.EXP_NAME)

        # compute input dataset checksum
        input_dataset_checksum = hashlib.md5(
            open(Constants.INPUT_FILE_PATH, "rb").read()
        ).hexdigest()

        # verify the logged parameters
        assert logged_params["input_dataset_checksum"] == input_dataset_checksum
        assert logged_params["sampling_style"] == sampling_style
        if sampling_ratio is not None:
            assert logged_params["sampling_ratio"] == str(sampling_ratio)
        if n_samples is not None:
            assert logged_params["n_samples"] == str(n_samples)
        if sampling_style == "random":
            assert logged_params["random_seed"] == str(random_seed)


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
            expected_exception_mssg = (
                f"Input file '{input_file_path}' is not a .jsonl file."
            )
        else:
            # Create two `.jsonl` files in the folder, folder as input is now invalid
            input_file_path = self._create_input_file(
                temp_dir, file_name="temp_test_1.jsonl"
            )
            input_file_path = self._create_input_file(
                temp_dir, file_name="temp_test_2.jsonl"
            )
            input_file_path = temp_dir
            expected_exception_mssg = (
                "More than one file in URI_FOLDER. Please specify a .jsonl URI_FILE or a URI_FOLDER "
                "with a single .jsonl file or an MLTable in input 'dataset'."
            )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

    @pytest.mark.skip("Dataset Sampler does not care if the jsonl file is valid or not.")
    def test_invalid_jsonl_file_as_dataset(self, temp_dir: str):
        """Test for invalid jsonl file as dataset."""
        # Create input file and expected exception message
        input_file_path = os.path.join(temp_dir, "invalid_data.jsonl")
        lines = [
            '{"name": "Person A", "age": 25}',
            'Invalid JSON Line',
            '{"name": "Person B", "age": 35}',
        ]
        with open(input_file_path, 'w') as file:
            for line in lines:
                file.write(line + '\n')

        expected_exception_mssg = (
            f"Input file '{input_file_path}' is not a valid .jsonl file."
        )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

    @pytest.mark.parametrize("dataset_type", ["mltable", "uri_file", "uri_folder"])
    def test_valid_dataset(self, temp_dir, dataset_type: str):
        """Test for valid input dataset."""
        # Create input file
        shutil.copy(Constants.INPUT_FILE_PATH, temp_dir)

        input_file_path = os.path.join(temp_dir, Constants.INPUT_FILE_PATH.split("/")[-1])
        if dataset_type == "uri_folder":
            input_file_path = temp_dir
        elif dataset_type == "mltable":
            paths = [
                {
                    "file": input_file_path,
                }
            ]
            mltable.from_json_lines_files(paths).save(temp_dir)

        # Run the script
        out_file_path = os.path.join(temp_dir, "output.jsonl")
        self._run_sampler_script(
            input_file_path, n_samples=5, output_dataset=out_file_path
        )

        # Verify the output file
        _verify_and_get_output_records(out_file_path, n_samples=5)

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
        else:
            expected_exception_mssg = (
                f"Invalid sampling_ratio: {float(sampling_ratio)}. Please specify float in (0, 1]."
            )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(input_file_path, sampling_ratio=sampling_ratio)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

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
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

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
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

    @pytest.mark.parametrize("sampling_style", ["haphazard"])
    def test_invalid_sampling_style(self, temp_dir: str, sampling_style: str):
        """Test for invalid sampling style."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        # the message below is the default message for invalid choice directly from argparse
        expected_exception_mssg = (
            f"argument --sampling_style: invalid choice: '{sampling_style}' (choose from 'head', 'tail', 'random')"
        )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(
                input_file_path, sampling_style=sampling_style, sampling_ratio=0.2
            )
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

    @pytest.mark.parametrize("sampling_style", ["head", "tail"])
    def test_random_seed_for_non_random_sampling(
        self, temp_dir: str, sampling_style: str
    ):
        """Test random seed for non random sampling."""
        # Create input file and expected exception message
        input_file_path = self._create_input_file(temp_dir, file_name="temp_test.jsonl")
        expected_exception_mssg = (
            f"Received random_seed: {0}, but it won't be used. It is used only when 'sampling_style' is 'random'."
        )

        # Run the script and verify the exception
        try:
            self._run_sampler_script(
                input_file_path, sampling_style=sampling_style, sampling_ratio=0.2
            )
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert expected_exception_mssg in exception_message

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

        :param dataset: The input dataset.
        :type dataset: str
        :param sampling_style: The sampling style, defaults to "head"
        :type sampling_style: str, optional
        :param sampling_ratio: The sampling ratio, defaults to None
        :type sampling_ratio: Union[float, None], optional
        :param n_samples: The number of samples, defaults to None
        :type n_samples: Union[int, None], optional
        :param random_seed: The random seed, defaults to 0
        :type random_seed: int, optional
        :param output_dataset: The output dataset, defaults to "out.jsonl"
        :type output_dataset: str, optional
        :return: None
        :rtype: NoneType
        """
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "components",
            "src",
            "dataset-sampler",
            "main.py",
        )
        args = [
            "python",
            f"{script_path}",
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
        _ = subprocess.check_output(
            args,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )
