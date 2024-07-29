# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision dataset adapters."""

import ast
import io

from abc import ABC, abstractmethod
from typing import Optional

from datasets import Dataset
from PIL import Image

from aml_benchmark.utils.logging import get_logger


logger = get_logger(__name__)


class VisionDatasetAdapter(ABC):
    """Abstract class for adapting HF vision datasets to internal format."""

    def __init__(self, dataset: Dataset):
        """Make adapter, storing relevant information from dataset."""
        pass

    @abstractmethod
    def get_label(self, instance):
        """Extract the instance's label as a string."""
        pass

    @abstractmethod
    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        pass

    def get_other_fields(self, instance):
        return {}


class Cifar10Adapter(VisionDatasetAdapter):
    """Adapter for Cifar10 HF dataset."""

    def __init__(self, dataset: Dataset):
        """Make adapter, storing relevant information from dataset."""
        self.label_feature = dataset.features["label"]

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return self.label_feature.int2str(instance["label"])

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["img"]


class Food101Adapter(VisionDatasetAdapter):
    """Adapter for Food101 HF dataset."""

    def __init__(self, dataset: Dataset):
        """Make adapter, storing relevant information from dataset."""
        self.label_feature = dataset.features["label"]

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return self.label_feature.int2str(instance["label"])

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["image"]


class PatchCamelyonAdapter(VisionDatasetAdapter):
    """Adapter for PatchCamelyon HF dataset."""

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return "unhealthy" if instance["label"] else "healthy"

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["image"]


class Resisc45Adapter(VisionDatasetAdapter):
    """Adapter for Resisc45 HF dataset."""

    def __init__(self, dataset: Dataset):
        """Make adapter, storing relevant information from dataset."""
        self.label_feature = dataset.features["label"]

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return self.label_feature.int2str(instance["label"])

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["image"]


class GTSRBAdapter(VisionDatasetAdapter):
    """Adapter for GTSRB HF dataset."""

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        # TODO(rdondera): update when dataset used in actual benchmark.
        return str(instance["ClassId"])

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return Image.open(io.BytesIO(instance["Path"]["bytes"]))


class MMMUAdapter(VisionDatasetAdapter):
    """Adapter for MMMU HF dataset."""

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return instance["answer"]

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["image_1"]

    def get_other_fields(self, instance):
        return {
            "question": instance["question"],
            "answer_options": "||".join(ast.literal_eval(instance["options"])),
            "more_images": [instance[f"image_{i}"] for i in range(2, 8) if instance[f"image_{i}"] is not None],
        }


class VQA2Adapter(VisionDatasetAdapter):
    """Adapter for the VQA V2 HF dataset."""

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return "||".join(set([a["answer"] for a in instance["answers"]]))

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["image"]

    def get_other_fields(self, instance):
        return {
            "question": instance["question"] + " <image 1>",
            "answer_options": "",
            "more_images": [],
        }


class VizWizAdapter(VisionDatasetAdapter):
    """Adapter for the VizWiz HF dataset."""

    def get_label(self, instance):
        """Extract the instance's label as a string."""
        return "||".join(set(instance["answers"]))

    def get_pil_image(self, instance):
        """Extract the instance's image as a PIL image."""
        return instance["image"]

    def get_other_fields(self, instance):
        return {
            "question": instance["question"] + " <image 1>",
            "answer_options": "",
            "more_images": [],
            "keep": instance["category"] != "unanswerable",
        }


class VisionDatasetAdapterFactory:
    """Factory for making vision dataset adapters based on dataset names."""

    @staticmethod
    def get_adapter(dataset: Dataset) -> Optional[VisionDatasetAdapter]:
        """Make vision adapter based on dataset name."""
        VISION_ADAPTERS_BY_DATASET_NAME = {
            "cifar10": Cifar10Adapter,
            "food101": Food101Adapter,
            "patch_camelyon": PatchCamelyonAdapter,
            "resisc45": Resisc45Adapter,
            "gtsrb": GTSRBAdapter,
            "mmmu": MMMUAdapter,
            "vq_av2": VQA2Adapter,
            "viz_wiz-vqa": VizWizAdapter,
        }

        # Select the adapter class based on the dataset name. If name not available or not recognized, do not make
        # an adapter.
        if VISION_ADAPTERS_BY_DATASET_NAME.get(getattr(dataset.info, "dataset_name", None)) is None:
            logger.info("Not making a vision adapter for dataset with info {}.".format(dataset.info))
            return None
        adapter_cls = VISION_ADAPTERS_BY_DATASET_NAME[dataset.info.dataset_name]

        return adapter_cls(dataset)
