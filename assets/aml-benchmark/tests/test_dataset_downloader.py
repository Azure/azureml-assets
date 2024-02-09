# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Dataset Downloader Component."""

from typing import Union, Optional
import os
import subprocess

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
import pytest
from datasets import get_dataset_config_names, get_dataset_split_names

from .test_utils import (
    load_yaml_pipeline,
    ML_CLIENT_SINGLETON,
    Constants,
    download_outputs,
    get_src_dir,
    assert_exception_mssg,
    run_command,
    assert_logged_params,
)


class TestDatasetDownloaderComponent:
    """Tests for dataset downloader component."""

    EXP_NAME = "dataset-downloader-test"

    @pytest.mark.parametrize(
        "dataset_name, configuration, split, script",
        [
            ("xquad", "xquad.en", "validation", None),
            ("xquad", "xquad.en,xquad.hi", "all", None),
            ("xquad", "all", "all", None),
            (None, "all", "test", Constants.MATH_DATASET_LOADER_SCRIPT),
        ],
    )
    def test_dataset_downloader_component(
        self,
        temp_dir: str,
        dataset_name: Union[str, None],
        configuration: Union[str, None],
        split: Union[str, None],
        script: Union[str, None],
    ) -> None:
        """Dataset Downloader component test."""
        ml_client = ML_CLIENT_SINGLETON.ml_client

        pipeline_job = self._get_pipeline_job(
            dataset_name,
            configuration,
            split,
            script,
            self.test_dataset_downloader_component.__name__,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        file_count = 1
        path = dataset_name if dataset_name else script
        if configuration == "all":
            file_count = len(get_dataset_config_names(path))
        elif split == "all":
            configs = configuration.split(",")
            file_count = sum(len(get_dataset_split_names(path, config)) for config in configs)
        self._verify_output(pipeline_job, temp_dir, file_count)
        assert_logged_params(
            pipeline_job.name,
            self.EXP_NAME,
            dataset_name=dataset_name,
            configuration=configuration,
            split=split,
            script=script,
        )

    def _get_pipeline_job(
        self,
        dataset_name: Union[str, None],
        configuration: Union[str, None],
        split: Union[str, None],
        script: Union[str, None],
        display_name: str,
    ) -> Job:
        """Get the pipeline job.

        :param dataset_name: Name of the dataset to download from HuggingFace.
        :param configuration: Specific sub-dataset of the HuggingFace dataset to download.
        :param split: Specific split of the HuggingFace dataset to download.
        :param script: Path to the data loader script.
        :param display_name: The display name for job.
        :return: The pipeline job.
        """
        pipeline_job = load_yaml_pipeline("dataset_downloader_pipeline.yaml")

        # set the pipeline inputs
        pipeline_job.inputs.dataset_name = dataset_name
        pipeline_job.inputs.configuration = configuration
        pipeline_job.inputs.split = split
        pipeline_job.inputs.script_path = Input(type="uri_file", path=script) if script else None
        pipeline_job.display_name = display_name

        return pipeline_job

    def _verify_output(self, job: Job, output_dir: str, file_count: int) -> None:
        """Verify the output.

        :param job: The job object.
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
        assert os.path.exists(output_dir)
        assert sum(len(files) for _, _, files in os.walk(output_dir)) == file_count


class TestDatasetDownloaderScript:
    """Tests for dataset downloader script."""

    @pytest.mark.parametrize(
        "dataset_name, split, script",
        [(None, "train", None), ("dataset", "test", "script")],
    )
    def test_invalid_input_combination(
        self,
        dataset_name: Union[str, None],
        split: Union[str, None],
        script: Union[str, None],
    ):
        """Test for invalid input combination."""
        if dataset_name and script:
            expected_exception_mssg = "Either 'dataset_name' or 'script' must be supplied; but not both."
        elif not (dataset_name or script):
            expected_exception_mssg = "Either 'dataset_name' or 'script' must be supplied."

        # Run the script and verify the exception
        try:
            self._run_downloader_script(dataset_name, None, split, script)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize(
        "dataset_name, configuration, split",
        [
            ("squad_v2", None, "test"),
            ("ai2_arc", None, "validation"),
            ("some_random_name", None, "test"),
        ],
    )
    def test_invalid_hf_dataset(
        self, dataset_name: str, configuration: Optional[str], split: str
    ):
        """Test for unsupported url file."""
        expected_exception_mssg = f"Split '{split}' not available for dataset '{dataset_name}' and config '{None}'."
        if dataset_name == "ai2_arc" and configuration is None:
            expected_exception_mssg = (
                f"Multiple configurations available for dataset '{dataset_name}'. Please specify either one of "
                f"the following: {get_dataset_config_names(dataset_name)} or 'all'."
            )
        elif dataset_name == "some_random_name":
            expected_exception_mssg = "FileNotFoundError: Couldn't find a dataset script at "

        # Run the script and verify the exception
        try:
            self._run_downloader_script(dataset_name, None, split, None)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, expected_exception_mssg)

    def _run_downloader_script(
        self,
        dataset_name: Union[str, None],
        configuration: Union[str, None],
        split: Union[str, None],
        script: Union[str, None],
        output_dataset: str = "out",
    ) -> None:
        """
        Run the dataset downloader script with the given arguments.

        :param dataset_name: The name of the dataset.
        :param configuration: The configuration of the dataset.
        :param split: The split of the dataset.
        :param script: Path to the data loader script.
        :param output_dataset: The output dataset file path, default is "out.jsonl".
        :return: None.
        """
        src_dir = get_src_dir()
        args = [
            f"cd {src_dir} &&",
            "python -m aml_benchmark.dataset_downloader.main",
            "--output_dataset",
            f"{output_dataset}",
            "--split",
            f"{split}",
        ]
        if dataset_name is not None:
            args.extend(["--dataset_name", f"{dataset_name}"])
        if configuration is not None:
            args.extend(["--configuration", f"{configuration}"])
        if script is not None:
            args.extend(["--script", f"{script}"])

        run_command(" ".join(args))
