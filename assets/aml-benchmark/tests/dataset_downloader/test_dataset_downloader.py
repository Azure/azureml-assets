# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for Dataset Downloader Component."""

from typing import Union, Optional
from collections import namedtuple
import io
import os

from azure.ai.ml.entities import Job
from azure.ai.ml import Input
import pytest
from datasets import get_dataset_config_names, get_dataset_split_names
from PIL import Image, ImageChops

from aml_benchmark.dataset_downloader.vision_dataset_adapter import VisionDatasetAdapterFactory
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


class TestDatasetDownloaderComponent:
    """Tests for dataset downloader component."""

    EXP_NAME = "dataset-downloader-test"

    @pytest.mark.parametrize(
        "dataset_name, configuration, split, script",
        [
            ("xquad", "xquad.en", "validation", None),
            ("xquad", "xquad.en,xquad.hi", "all", None),
            ("xquad", "all", "all", None),
            ("cifar10", "all", "test", None),
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
        ml_client = get_mlclient()

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
        except RuntimeError as e:
            exception_message = str(e)
            assert_exception_mssg(exception_message, expected_exception_mssg)

    @pytest.mark.parametrize(
        "dataset_name, configuration, split",
        [
            ("squad_v2", None, "test"),
            ("winogrande", None, "validation"),
            ("some_random_name", None, "test"),
        ],
    )
    def test_invalid_hf_dataset(
        self, dataset_name: str, configuration: Optional[str], split: str
    ):
        """Test for unsupported url file."""
        expected_exception_mssg = f"Split '{split}' not available for dataset '{dataset_name}' and config '{None}'."
        if dataset_name == "winogrande" and configuration is None:
            expected_exception_mssg = (
                f"Multiple configurations available for dataset '{dataset_name}'. Please specify either one of "
                f"the following: {get_dataset_config_names(dataset_name)} or 'all'."
            )
        elif dataset_name == "some_random_name":
            expected_exception_mssg = f"FileNotFoundError: Dataset '{dataset_name}' doesn't exist on the Hub"

        # Run the script and verify the exception
        try:
            self._run_downloader_script(dataset_name, None, split, None)
        except RuntimeError as e:
            exception_message = str(e)
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


class TestVisionDatasetAdapters:
    class MockDataset:
        def __init__(self, name, label_mapping):
            self.info = namedtuple("dataset_info", "dataset_name")(name)
            self.features = {
                "label": namedtuple("label_info", "int2str")(
                    lambda x: label_mapping[x]
                )
            }

    def test_no_vision_adapter(self):
        dataset = self.MockDataset("glue", {})

        adapter = VisionDatasetAdapterFactory.get_adapter(dataset)

        assert adapter is None

    def test_cifar10_adapter(self):
        dataset = self.MockDataset("cifar10", {0: "airplane"})
        label = 0
        image = Image.new("RGB", (640, 480))
        instance = {
            "label": label,
            "img": image,
        }

        adapter = VisionDatasetAdapterFactory.get_adapter(dataset)

        assert adapter.get_label(instance) == "airplane"
        assert adapter.get_pil_image(instance) == image

    def test_food101_adapter(self):
        dataset = self.MockDataset("food101", {13: "beignets"})
        label = 13
        image = Image.new("RGB", (640, 480))
        instance = {
            "label": label,
            "image": image,
        }

        adapter = VisionDatasetAdapterFactory.get_adapter(dataset)

        assert adapter.get_label(instance) == "beignets"
        assert adapter.get_pil_image(instance) == image

    @pytest.mark.parametrize("label", [0, 1])
    def test_patch_camelyon_adapter(self, label):
        dataset = self.MockDataset("patch_camelyon", {})
        image = Image.new("RGB", (640, 480))
        instance = {
            "label": label == 1,
            "image": image,
        }

        adapter = VisionDatasetAdapterFactory.get_adapter(dataset)

        assert adapter.get_label(instance) == ("healthy" if label == 0 else "unhealthy")
        assert adapter.get_pil_image(instance) == image

    def test_resisc45_adapter(self):
        dataset = self.MockDataset(
            "resisc45",
            {
                0: "airport",
                1: "bridge",
                2: "commercial_area",
                3: "desert",
                4: "forest",
                5: "freeway",
                6: "harbor",
                7: "lake",
                8: "palace",
                9: "railway_station",
            }
        )
        label = 5
        image = Image.new("RGB", (640, 480))
        instance = {
            "label": label,
            "image": image,
            "image_id": "423-153",
        }

        adapter = VisionDatasetAdapterFactory.get_adapter(dataset)

        assert adapter.get_label(instance) == "freeway"
        assert adapter.get_pil_image(instance) == image

    def test_gtsrb_adapter(self):
        dataset = self.MockDataset(
            "gtsrb",
            {
                c: str(c)
                for c in range(42)
            }
        )
        label = 4
        image = Image.new("RGB", (640, 480))

        # Save image bytes to memory buffer.
        buffer = io.BytesIO()
        image.save(buffer, "png")

        instance = {
            "Width": image.width,
            "Height": image.height,
            "Roi.X1": 0,
            "Roi.Y1": 0,
            "Roi.X2": image.width - 1,
            "Roi.Y2": image.height - 1,
            "ClassId": label,
            "Path": {
                "bytes": buffer.getvalue(),
                "path": "a/b/c.png",
            }
        }

        adapter = VisionDatasetAdapterFactory.get_adapter(dataset)

        assert adapter.get_label(instance) == str(label)
        assert not ImageChops.difference(adapter.get_pil_image(instance), image).getbbox()
