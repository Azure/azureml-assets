# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Vision dataset adapters."""

import io

from abc import ABC, abstractmethod

from datasets import Dataset
from PIL import Image


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


class VisionDatasetAdapterFactory:
    """Factory for making vision adapters based on dataset names."""

    @staticmethod
    def get_adapter(dataset: Dataset) -> VisionDatasetAdapter:
        """Make vision adapter based on dataset name."""
        VISION_ADAPTERS_BY_DATASET_NAME = {
            "cifar10": Cifar10Adapter,
            "food101": Food101Adapter,
            "patch_camelyon": PatchCamelyonAdapter,
            "resisc45": Resisc45Adapter,
            "gtsrb": GTSRBAdapter,
        }

        adapter_cls = VISION_ADAPTERS_BY_DATASET_NAME.get(dataset.info.dataset_name)
        if adapter_cls is None:
            return None

        return adapter_cls(dataset)
