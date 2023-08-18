# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Dataset Downloader Component."""

from typing import Union
import os
import subprocess

from azure.ai.ml.entities import Job
import pytest
from datasets import get_dataset_config_names, get_dataset_split_names

from utils import (
    load_yaml_pipeline,
    get_mlclient,
    Constants,
    download_outputs,
    get_mlflow_logged_params,
    get_src_dir,
    assert_exception_mssg,
    run_command,
)


class TestDatasetDownloaderComponent:
    """Tests for dataset downloader component."""

    EXP_NAME = "dataset-downloader-test"

    @pytest.mark.parametrize(
        "dataset_name, configuration, split, url",
        [
            ("xquad", "xquad.en", "validation", None),
            ("xquad", "xquad.en", "all", None),
            ("xquad", "all", "all", None),
            (None, None, None, Constants.REMOTE_FILE_URL),
        ],
    )
    def test_dataset_downloader_component(
        self,
        temp_dir: str,
        dataset_name: Union[str, None],
        configuration: Union[str, None],
        split: Union[str, None],
        url: Union[str, None],
    ) -> None:
        """Dataset Downloader component test."""
        ml_client = get_mlclient()

        pipeline_job = self._get_pipeline_job(
            dataset_name,
            configuration,
            split,
            url,
            self.test_dataset_downloader_component.__name__,
        )

        # submit the pipeline job
        pipeline_job = ml_client.create_or_update(
            pipeline_job, experiment_name=self.EXP_NAME
        )
        ml_client.jobs.stream(pipeline_job.name)
        print(pipeline_job)

        file_count = 1
        if configuration == "all":
            file_count = len(get_dataset_config_names(dataset_name))
        elif split == "all":
            file_count = len(get_dataset_split_names(dataset_name, configuration))
        self._verify_output(pipeline_job, temp_dir, file_count)
        self._verify_logged_params(
            pipeline_job.name, dataset_name, configuration, split, url
        )

    def _get_pipeline_job(
        self,
        dataset_name: Union[str, None],
        configuration: Union[str, None],
        split: Union[str, None],
        url: Union[str, None],
        display_name: str,
    ) -> Job:
        """Get the pipeline job.

        :param dataset_name: Name of the dataset to download from HuggingFace.
        :param configuration: Specific sub-dataset of the HuggingFace dataset to download.
        :param split: Specific split of the HuggingFace dataset to download.
        :param display_name: The display name for job.
        :return: The pipeline job.
        """
        pipeline_job = load_yaml_pipeline("dataset_downloader_pipeline.yaml")

        # set the pipeline inputs
        pipeline_job.inputs.dataset_name = dataset_name
        pipeline_job.inputs.configuration = configuration
        pipeline_job.inputs.split = split
        pipeline_job.inputs.url = url
        pipeline_job.display_name = display_name

        return pipeline_job

    def _verify_output(
        self,
        job: Job,
        output_dir: str,
        file_count: int
    ) -> None:
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

    def _verify_logged_params(
        self,
        job_name: str,
        dataset_name: Union[str, None],
        configuration: Union[str, None],
        split: Union[str, None],
        url: Union[str, None],
    ) -> None:
        """Verify the logged parameters.

        :param job_name: The job name.
        :param dataset_name: Name of the dataset.
        :param configuration: Specific sub-dataset of the HuggingFace.
        :param split: Specific split of the HuggingFace dataset.
        :param url: The url that was used to download the dataset from.
        :return: None.
        """
        logged_params = get_mlflow_logged_params(job_name, self.EXP_NAME)

        # verify the logged parameters
        if dataset_name is not None:
            assert logged_params["dataset_name"] == dataset_name
        if configuration is not None:
            assert logged_params["configuration"] == configuration
        if split is not None:
            assert logged_params["split"] == split
        if url is not None:
            assert logged_params["url"] == url


class TestDatasetDownloaderScript:
    """Tests for dataset downloader script."""

    @pytest.mark.parametrize(
        "dataset_name, split, url",
        [("dataset", None, None), (None, "train", None), ("dataset", "test", "url")],
    )
    def test_invalid_input_combination(
        self,
        dataset_name: Union[str, None],
        split: Union[str, None],
        url: Union[str, None],
    ):
        """Test for invalid input combination."""
        if dataset_name and split and url:
            expected_exception_mssg = "Either 'dataset_name' with 'split', or 'url' must be supplied; but not both."
        else:
            expected_exception_mssg = (
                "Either 'dataset_name' with 'split', or 'url' must be supplied."
            )

        # Run the script and verify the exception
        try:
            self._run_downloader_script(dataset_name, None, split, url)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, expected_exception_mssg)

    def test_unsupported_url_file(self):
        """Test for unsupported url file."""
        url = "no_link.zzz"
        expected_exception_mssg = (
            f"File extension '{url.split('.')[-1]}' not supported."
        )

        # Run the script and verify the exception
        try:
            self._run_downloader_script(None, None, None, url)
        except subprocess.CalledProcessError as e:
            exception_message = e.output.strip()
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize("dataset_name, split", [("squad_v2", "test")])
    def test_invalid_hf_dataset(self, dataset_name: str, split: str):
        """Test for unsupported url file."""
        expected_exception_mssg = f"Split '{split}' not available for dataset '{dataset_name}' and config '{None}'."
        if split is None:
            expected_exception_mssg = "Split can't be None."

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
        url: Union[str, None],
        output_dataset: str = "out",
    ) -> None:
        """
        Run the dataset downloader script with the given arguments.

        :param dataset_name: The name of the dataset.
        :param configuration: The configuration of the dataset.
        :param split: The split of the dataset.
        :param url: The url of the dataset.
        :param output_dataset: The output dataset file path, default is "out.jsonl".
        :return: None.
        """
        src_dir = get_src_dir()
        args = [
            f"cd {src_dir} &&",
            "python -m dataset_downloader.main",
            "--output_dataset",
            f"{output_dataset}",
        ]
        if dataset_name is not None:
            args.extend(["--dataset_name", f"{dataset_name}"])
        if configuration is not None:
            args.extend(["--configuration", f"{configuration}"])
        if split is not None:
            args.extend(["--split", f"{split}"])
        if url is not None:
            args.extend(["--url", f"{url}"])

        run_command(" ".join(args))
