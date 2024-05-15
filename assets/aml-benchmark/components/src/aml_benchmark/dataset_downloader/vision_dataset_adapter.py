import io

from abc import ABC, abstractmethod

from datasets import Dataset
from PIL import Image


from aml_benchmark.utils.logging import get_logger
logger = get_logger(__name__)


class VisionDatasetAdapter(ABC):
    def __init__(self, dataset: Dataset):
        pass

    @abstractmethod
    def get_label(self, instance):
        pass

    @abstractmethod
    def get_pil_image(self, instance):
        pass


class Food101Adapter(VisionDatasetAdapter):
    def __init__(self, dataset: Dataset):
        self.label_feature = dataset.features["label"]

    def get_label(self, instance):
        return self.label_feature.int2str(instance["label"])

    def get_pil_image(self, instance):
        return instance["image"]


class PatchCamelyonAdapter(VisionDatasetAdapter):
    def get_label(self, instance):
        return "unhealthy" if instance["label"] else "healthy"

    def get_pil_image(self, instance):
        return instance["image"]


class Resisc45Adapter(VisionDatasetAdapter):
    def __init__(self, dataset: Dataset):
        self.label_feature = dataset.features["label"]

    def get_label(self, instance):
        return self.label_feature.int2str(instance["label"])

    def get_pil_image(self, instance):
        return instance["image"]


class GTSRBAdapter(VisionDatasetAdapter):
    def get_label(self, instance):
        # TODO(rdondera): update when dataset used in actual benchmark.
        return str(instance["ClassId"])

    def get_pil_image(self, instance):
        return Image.open(io.BytesIO(instance["Path"]["bytes"]))


class VisionDatasetAdapterFactory:
    @staticmethod
    def get_adapter(dataset: Dataset) -> VisionDatasetAdapter:
        VISION_ADAPTERS_BY_DATASET_NAME = {
            # "food101": Food101Adapter,
            # "1aurent/PatchCamelyon": PatchCamelyonAdapter,
            # "timm/resisc45": Resisc45Adapter,
            # "bazyl/GTSRB": GTSRBAdapter,
            "food101": Food101Adapter,
            "patch_camelyon": PatchCamelyonAdapter,
            "resisc45": Resisc45Adapter,
            "gtsrb": GTSRBAdapter,
        }

        adapter_cls = VISION_ADAPTERS_BY_DATASET_NAME.get(dataset.info.dataset_name)
        if adapter_cls is None:
            return None

        return adapter_cls(dataset)
