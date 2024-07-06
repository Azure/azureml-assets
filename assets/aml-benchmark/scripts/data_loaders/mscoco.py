# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Data Loading Script for MSCOCO.

Adapted from https://github.com/shunk031/huggingface-datasets_MSCOCO/blob/main/MSCOCO.py to only download validation
images.
"""

import abc
import json
import logging
import os
from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    Final,
    Iterator,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
    get_args,
)

import datasets as ds
import numpy as np
from datasets.data_files import DataFilesDict
from PIL import Image
from PIL.Image import Image as PilImage
from pycocotools import mask as cocomask
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

JsonDict = Dict[str, Any]
ImageId = int
AnnotationId = int
LicenseId = int
CategoryId = int
Bbox = Tuple[float, float, float, float]

MscocoSplits = Literal["train", "val", "test"]

KEYPOINT_STATE: Final[List[str]] = ["unknown", "invisible", "visible"]


_CITATION = """
"""

_DESCRIPTION = """
"""

_HOMEPAGE = """
"""

_LICENSE = "https://creativecommons.org/licenses/by/4.0/legalcode"

_URLS = {
    "2014": {
        "images": {
            "train": "http://images.cocodataset.org/zips/train2014.zip",
            "validation": "http://images.cocodataset.org/zips/val2014.zip",
            "test": "http://images.cocodataset.org/zips/test2014.zip",
        },
        "annotations": {
            "train_validation": "http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
            "test_image_info": "http://images.cocodataset.org/annotations/image_info_test2014.zip",
        },
    },
    "2015": {
        "images": {
            "test": "http://images.cocodataset.org/zips/test2015.zip",
        },
        "annotations": {
            "test_image_info": "http://images.cocodataset.org/annotations/image_info_test2015.zip",
        },
    },
    "2017": {
        "images": {
            # "train": "http://images.cocodataset.org/zips/train2017.zip",
            "validation": "http://images.cocodataset.org/zips/val2017.zip",
            # "test": "http://images.cocodataset.org/zips/test2017.zip",
            # "unlabeled": "http://images.cocodataset.org/zips/unlabeled2017.zip",
        },
        "annotations": {
            "train_validation": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
            "stuff_train_validation": "http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip",
            "panoptic_train_validation": "http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip",
            "test_image_info": "http://images.cocodataset.org/annotations/image_info_test2017.zip",
            "unlabeled": "http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip",
        },
    },
}

CATEGORIES: Final[List[str]] = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]

SUPER_CATEGORIES: Final[List[str]] = [
    "person",
    "vehicle",
    "outdoor",
    "animal",
    "accessory",
    "sports",
    "kitchen",
    "food",
    "furniture",
    "electronic",
    "appliance",
    "indoor",
]


@dataclass
class AnnotationInfo(object):
    description: str
    url: str
    version: str
    year: str
    contributor: str
    date_created: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "AnnotationInfo":
        return cls(**json_dict)


@dataclass
class LicenseData(object):
    url: str
    license_id: LicenseId
    name: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "LicenseData":
        return cls(
            license_id=json_dict["id"],
            url=json_dict["url"],
            name=json_dict["name"],
        )


@dataclass
class ImageData(object):
    image_id: ImageId
    license_id: LicenseId
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "ImageData":
        return cls(
            image_id=json_dict["id"],
            license_id=json_dict["license"],
            file_name=json_dict["file_name"],
            coco_url=json_dict["coco_url"],
            height=json_dict["height"],
            width=json_dict["width"],
            date_captured=json_dict["date_captured"],
            flickr_url=json_dict["flickr_url"],
        )

    @property
    def shape(self) -> Tuple[int, int]:
        return (self.height, self.width)


@dataclass
class CategoryData(object):
    category_id: int
    name: str
    supercategory: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CategoryData":
        return cls(
            category_id=json_dict["id"],
            name=json_dict["name"],
            supercategory=json_dict["supercategory"],
        )


@dataclass
class AnnotationData(object):
    annotation_id: AnnotationId
    image_id: ImageId


@dataclass
class CaptionsAnnotationData(AnnotationData):
    caption: str

    @classmethod
    def from_dict(cls, json_dict: JsonDict) -> "CaptionsAnnotationData":
        return cls(
            annotation_id=json_dict["id"],
            image_id=json_dict["image_id"],
            caption=json_dict["caption"],
        )


class UncompressedRLE(TypedDict):
    counts: List[int]
    size: Tuple[int, int]


class CompressedRLE(TypedDict):
    counts: bytes
    size: Tuple[int, int]


@dataclass
class InstancesAnnotationData(AnnotationData):
    segmentation: Union[np.ndarray, CompressedRLE]
    area: float
    iscrowd: bool
    bbox: Tuple[float, float, float, float]
    category_id: int

    @classmethod
    def compress_rle(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> CompressedRLE:
        if iscrowd:
            rle = cocomask.frPyObjects(segmentation, h=height, w=width)
        else:
            rles = cocomask.frPyObjects(segmentation, h=height, w=width)
            rle = cocomask.merge(rles)

        return rle  # type: ignore

    @classmethod
    def rle_segmentation_to_binary_mask(
        cls, segmentation, iscrowd: bool, height: int, width: int
    ) -> np.ndarray:
        rle = cls.compress_rle(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return cocomask.decode(rle)  # type: ignore

    @classmethod
    def rle_segmentation_to_mask(
        cls,
        segmentation: Union[List[List[float]], UncompressedRLE],
        iscrowd: bool,
        height: int,
        width: int,
    ) -> np.ndarray:
        binary_mask = cls.rle_segmentation_to_binary_mask(
            segmentation=segmentation, iscrowd=iscrowd, height=height, width=width
        )
        return binary_mask * 255

    @classmethod
    def from_dict(
        cls,
        json_dict: JsonDict,
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
    ) -> "InstancesAnnotationData":
        segmentation = json_dict["segmentation"]
        image_id = json_dict["image_id"]
        image_data = images[image_id]
        iscrowd = bool(json_dict["iscrowd"])

        segmentation_mask = (
            cls.rle_segmentation_to_mask(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
            if decode_rle
            else cls.compress_rle(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
        )
        return cls(
            #
            # for AnnotationData
            #
            annotation_id=json_dict["id"],
            image_id=image_id,
            #
            # for InstancesAnnotationData
            #
            segmentation=segmentation_mask,  # type: ignore
            area=json_dict["area"],
            iscrowd=iscrowd,
            bbox=json_dict["bbox"],
            category_id=json_dict["category_id"],
        )


@dataclass
class PersonKeypoint(object):
    x: int
    y: int
    v: int
    state: str


@dataclass
class PersonKeypointsAnnotationData(InstancesAnnotationData):
    num_keypoints: int
    keypoints: List[PersonKeypoint]

    @classmethod
    def v_keypoint_to_state(cls, keypoint_v: int) -> str:
        return KEYPOINT_STATE[keypoint_v]

    @classmethod
    def get_person_keypoints(
        cls, flatten_keypoints: List[int], num_keypoints: int
    ) -> List[PersonKeypoint]:
        keypoints_x = flatten_keypoints[0::3]
        keypoints_y = flatten_keypoints[1::3]
        keypoints_v = flatten_keypoints[2::3]
        assert len(keypoints_x) == len(keypoints_y) == len(keypoints_v)

        keypoints = [
            PersonKeypoint(x=x, y=y, v=v, state=cls.v_keypoint_to_state(v))
            for x, y, v in zip(keypoints_x, keypoints_y, keypoints_v)
        ]
        assert len([kp for kp in keypoints if kp.state != "unknown"]) == num_keypoints
        return keypoints

    @classmethod
    def from_dict(
        cls,
        json_dict: JsonDict,
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
    ) -> "PersonKeypointsAnnotationData":
        segmentation = json_dict["segmentation"]
        image_id = json_dict["image_id"]
        image_data = images[image_id]
        iscrowd = bool(json_dict["iscrowd"])

        segmentation_mask = (
            cls.rle_segmentation_to_mask(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
            if decode_rle
            else cls.compress_rle(
                segmentation=segmentation,
                iscrowd=iscrowd,
                height=image_data.height,
                width=image_data.width,
            )
        )
        flatten_keypoints = json_dict["keypoints"]
        num_keypoints = json_dict["num_keypoints"]
        keypoints = cls.get_person_keypoints(flatten_keypoints, num_keypoints)

        return cls(
            #
            # for AnnotationData
            #
            annotation_id=json_dict["id"],
            image_id=image_id,
            #
            # for InstancesAnnotationData
            #
            segmentation=segmentation_mask,  # type: ignore
            area=json_dict["area"],
            iscrowd=iscrowd,
            bbox=json_dict["bbox"],
            category_id=json_dict["category_id"],
            #
            # PersonKeypointsAnnotationData
            #
            num_keypoints=num_keypoints,
            keypoints=keypoints,
        )


class LicenseDict(TypedDict):
    license_id: LicenseId
    name: str
    url: str


class BaseExample(TypedDict):
    image_id: ImageId
    image: PilImage
    file_name: str
    coco_url: str
    height: int
    width: int
    date_captured: str
    flickr_url: str
    license_id: LicenseId
    license: LicenseDict


class CaptionAnnotationDict(TypedDict):
    annotation_id: AnnotationId
    caption: str


class CaptionExample(BaseExample):
    annotations: List[CaptionAnnotationDict]


class CategoryDict(TypedDict):
    category_id: CategoryId
    name: str
    supercategory: str


class InstanceAnnotationDict(TypedDict):
    annotation_id: AnnotationId
    area: float
    bbox: Bbox
    image_id: ImageId
    category_id: CategoryId
    category: CategoryDict
    iscrowd: bool
    segmentation: np.ndarray


class InstanceExample(BaseExample):
    annotations: List[InstanceAnnotationDict]


class KeypointDict(TypedDict):
    x: int
    y: int
    v: int
    state: str


class PersonKeypointAnnotationDict(InstanceAnnotationDict):
    num_keypoints: int
    keypoints: List[KeypointDict]


class PersonKeypointExample(BaseExample):
    annotations: List[PersonKeypointAnnotationDict]


class MsCocoProcessor(object, metaclass=abc.ABCMeta):
    def load_image(self, image_path: str) -> PilImage:
        return Image.open(image_path)

    def load_annotation_json(self, ann_file_path: str) -> JsonDict:
        logger.info(f"Load annotation json from {ann_file_path}")
        with open(ann_file_path, "r") as rf:
            ann_json = json.load(rf)
        return ann_json

    def load_licenses_data(
        self, license_dicts: List[JsonDict]
    ) -> Dict[LicenseId, LicenseData]:
        licenses = {}
        for license_dict in license_dicts:
            license_data = LicenseData.from_dict(license_dict)
            licenses[license_data.license_id] = license_data
        return licenses

    def load_images_data(
        self,
        image_dicts: List[JsonDict],
        tqdm_desc: str = "Load images",
    ) -> Dict[ImageId, ImageData]:
        images = {}
        for image_dict in tqdm(image_dicts, desc=tqdm_desc):
            image_data = ImageData.from_dict(image_dict)
            images[image_data.image_id] = image_data
        return images

    def load_categories_data(
        self,
        category_dicts: List[JsonDict],
        tqdm_desc: str = "Load categories",
    ) -> Dict[CategoryId, CategoryData]:
        categories = {}
        for category_dict in tqdm(category_dicts, desc=tqdm_desc):
            category_data = CategoryData.from_dict(category_dict)
            categories[category_data.category_id] = category_data
        return categories

    def get_features_base_dict(self):
        return {
            "image_id": ds.Value("int64"),
            "image": ds.Image(),
            "file_name": ds.Value("string"),
            "coco_url": ds.Value("string"),
            "height": ds.Value("int32"),
            "width": ds.Value("int32"),
            "date_captured": ds.Value("string"),
            "flickr_url": ds.Value("string"),
            "license_id": ds.Value("int32"),
            "license": {
                "url": ds.Value("string"),
                "license_id": ds.Value("int8"),
                "name": ds.Value("string"),
            },
        }

    @abc.abstractmethod
    def get_features(self, *args, **kwargs) -> ds.Features:
        raise NotImplementedError

    @abc.abstractmethod
    def load_data(self, ann_dicts: List[JsonDict], tqdm_desc: str = "", **kwargs):
        assert tqdm_desc != "", "tqdm_desc must be provided."
        raise NotImplementedError

    @abc.abstractmethod
    def generate_examples(
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[CaptionsAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        **kwargs,
    ):
        raise NotImplementedError


class CaptionsProcessor(MsCocoProcessor):
    def get_features(self, *args, **kwargs) -> ds.Features:
        features_dict = self.get_features_base_dict()
        annotations = ds.Sequence(
            {
                "annotation_id": ds.Value("int64"),
                "image_id": ds.Value("int64"),
                "caption": ds.Value("string"),
            }
        )
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_data(
        self,
        ann_dicts: List[JsonDict],
        tqdm_desc: str = "Load captions data",
        **kwargs,
    ) -> Dict[ImageId, List[CaptionsAnnotationData]]:
        annotations = defaultdict(list)
        for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
            ann_data = CaptionsAnnotationData.from_dict(ann_dict)
            annotations[ann_data.image_id].append(ann_data)
        return annotations

    def generate_examples(
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[CaptionsAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        **kwargs,
    ) -> Iterator[Tuple[int, CaptionExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            assert len(image_anns) > 0

            image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example = asdict(image_data)
            example["image"] = image
            example["license"] = asdict(licenses[image_data.license_id])

            example["annotations"] = []
            for ann in image_anns:
                example["annotations"].append(asdict(ann))

            yield idx, example  # type: ignore


class InstancesProcessor(MsCocoProcessor):
    def get_features_instance_dict(self, decode_rle: bool):
        segmentation_feature = (
            ds.Image()
            if decode_rle
            else {
                "counts": ds.Sequence(ds.Value("int64")),
                "size": ds.Sequence(ds.Value("int32")),
            }
        )
        return {
            "annotation_id": ds.Value("int64"),
            "image_id": ds.Value("int64"),
            "segmentation": segmentation_feature,
            "area": ds.Value("float32"),
            "iscrowd": ds.Value("bool"),
            "bbox": ds.Sequence(ds.Value("float32"), length=4),
            "category_id": ds.Value("int32"),
            "category": {
                "category_id": ds.Value("int32"),
                "name": ds.ClassLabel(
                    num_classes=len(CATEGORIES),
                    names=CATEGORIES,
                ),
                "supercategory": ds.ClassLabel(
                    num_classes=len(SUPER_CATEGORIES),
                    names=SUPER_CATEGORIES,
                ),
            },
        }

    def get_features(self, decode_rle: bool) -> ds.Features:
        features_dict = self.get_features_base_dict()
        annotations = ds.Sequence(
            self.get_features_instance_dict(decode_rle=decode_rle)
        )
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_data(  # type: ignore[override]
        self,
        ann_dicts: List[JsonDict],
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
        tqdm_desc: str = "Load instances data",
    ) -> Dict[ImageId, List[InstancesAnnotationData]]:
        annotations = defaultdict(list)
        ann_dicts = sorted(ann_dicts, key=lambda d: d["image_id"])

        for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
            ann_data = InstancesAnnotationData.from_dict(
                ann_dict, images=images, decode_rle=decode_rle
            )
            annotations[ann_data.image_id].append(ann_data)

        return annotations

    def generate_examples(  # type: ignore[override]
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[InstancesAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        categories: Dict[CategoryId, CategoryData],
    ) -> Iterator[Tuple[int, InstanceExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            if len(image_anns) < 1:
                logger.warning(f"No annotation found for image id: {image_id}.")
                continue

            image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example = asdict(image_data)
            example["image"] = image
            example["license"] = asdict(licenses[image_data.license_id])

            example["annotations"] = []
            for ann in image_anns:
                ann_dict = asdict(ann)
                category = categories[ann.category_id]
                ann_dict["category"] = asdict(category)
                example["annotations"].append(ann_dict)

            yield idx, example  # type: ignore


class PersonKeypointsProcessor(InstancesProcessor):
    def get_features(self, decode_rle: bool) -> ds.Features:
        features_dict = self.get_features_base_dict()
        features_instance_dict = self.get_features_instance_dict(decode_rle=decode_rle)
        features_instance_dict.update(
            {
                "keypoints": ds.Sequence(
                    {
                        "state": ds.Value("string"),
                        "x": ds.Value("int32"),
                        "y": ds.Value("int32"),
                        "v": ds.Value("int32"),
                    }
                ),
                "num_keypoints": ds.Value("int32"),
            }
        )
        annotations = ds.Sequence(features_instance_dict)
        features_dict.update({"annotations": annotations})
        return ds.Features(features_dict)

    def load_data(  # type: ignore[override]
        self,
        ann_dicts: List[JsonDict],
        images: Dict[ImageId, ImageData],
        decode_rle: bool,
        tqdm_desc: str = "Load person keypoints data",
    ) -> Dict[ImageId, List[PersonKeypointsAnnotationData]]:
        annotations = defaultdict(list)
        ann_dicts = sorted(ann_dicts, key=lambda d: d["image_id"])

        for ann_dict in tqdm(ann_dicts, desc=tqdm_desc):
            ann_data = PersonKeypointsAnnotationData.from_dict(
                ann_dict, images=images, decode_rle=decode_rle
            )
            annotations[ann_data.image_id].append(ann_data)
        return annotations

    def generate_examples(  # type: ignore[override]
        self,
        image_dir: str,
        images: Dict[ImageId, ImageData],
        annotations: Dict[ImageId, List[PersonKeypointsAnnotationData]],
        licenses: Dict[LicenseId, LicenseData],
        categories: Dict[CategoryId, CategoryData],
    ) -> Iterator[Tuple[int, PersonKeypointExample]]:
        for idx, image_id in enumerate(images.keys()):
            image_data = images[image_id]
            image_anns = annotations[image_id]

            if len(image_anns) < 1:
                # If there are no persons in the image,
                # no keypoint annotations will be assigned.
                continue

            image = self.load_image(
                image_path=os.path.join(image_dir, image_data.file_name),
            )
            example = asdict(image_data)
            example["image"] = image
            example["license"] = asdict(licenses[image_data.license_id])

            example["annotations"] = []
            for ann in image_anns:
                ann_dict = asdict(ann)
                category = categories[ann.category_id]
                ann_dict["category"] = asdict(category)
                example["annotations"].append(ann_dict)

            yield idx, example  # type: ignore


class MsCocoConfig(ds.BuilderConfig):
    YEARS: Tuple[int, ...] = (
        2014,
        2017,
    )
    TASKS: Tuple[str, ...] = (
        "captions",
        "instances",
        "person_keypoints",
    )

    def __init__(
        self,
        year: int,
        coco_task: Union[str, Sequence[str]],
        version: Optional[Union[ds.Version, str]],
        decode_rle: bool = False,
        data_dir: Optional[str] = None,
        data_files: Optional[DataFilesDict] = None,
        description: Optional[str] = None,
    ) -> None:
        super().__init__(
            name=self.config_name(year=year, task=coco_task),
            version=version,
            data_dir=data_dir,
            data_files=data_files,
            description=description,
        )
        self._check_year(year)
        self._check_task(coco_task)

        self._year = year
        self._task = coco_task
        self.processor = self.get_processor()
        self.decode_rle = decode_rle

    def _check_year(self, year: int) -> None:
        assert year in self.YEARS, year

    def _check_task(self, task: Union[str, Sequence[str]]) -> None:
        if isinstance(task, str):
            assert task in self.TASKS, task
        elif isinstance(task, list) or isinstance(task, tuple):
            for t in task:
                assert t, task
        else:
            raise ValueError(f"Invalid task: {task}")

    @property
    def year(self) -> int:
        return self._year

    @property
    def task(self) -> str:
        if isinstance(self._task, str):
            return self._task
        elif isinstance(self._task, list) or isinstance(self._task, tuple):
            return "-".join(sorted(self._task))
        else:
            raise ValueError(f"Invalid task: {self._task}")

    def get_processor(self) -> MsCocoProcessor:
        if self.task == "captions":
            return CaptionsProcessor()
        elif self.task == "instances":
            return InstancesProcessor()
        elif self.task == "person_keypoints":
            return PersonKeypointsProcessor()
        else:
            raise ValueError(f"Invalid task: {self.task}")

    @classmethod
    def config_name(cls, year: int, task: Union[str, Sequence[str]]) -> str:
        if isinstance(task, str):
            return f"{year}-{task}"
        elif isinstance(task, list) or isinstance(task, tuple):
            task = "-".join(task)
            return f"{year}-{task}"
        else:
            raise ValueError(f"Invalid task: {task}")


def dataset_configs(year: int, version: ds.Version) -> List[MsCocoConfig]:
    return [
        MsCocoConfig(
            year=year,
            coco_task="captions",
            version=version,
        ),
        MsCocoConfig(
            year=year,
            coco_task="instances",
            version=version,
        ),
        MsCocoConfig(
            year=year,
            coco_task="person_keypoints",
            version=version,
        ),
        # MsCocoConfig(
        #     year=year,
        #     coco_task=("captions", "instances"),
        #     version=version,
        # ),
        # MsCocoConfig(
        #     year=year,
        #     coco_task=("captions", "person_keypoints"),
        #     version=version,
        # ),
    ]


def configs_2014(version: ds.Version) -> List[MsCocoConfig]:
    return dataset_configs(year=2014, version=version)


def configs_2017(version: ds.Version) -> List[MsCocoConfig]:
    return dataset_configs(year=2017, version=version)


class MsCocoDataset(ds.GeneratorBasedBuilder):
    VERSION = ds.Version("1.0.0")
    BUILDER_CONFIG_CLASS = MsCocoConfig
    BUILDER_CONFIGS = configs_2014(version=VERSION) + configs_2017(version=VERSION)

    @property
    def year(self) -> int:
        config: MsCocoConfig = self.config  # type: ignore
        return config.year

    @property
    def task(self) -> str:
        config: MsCocoConfig = self.config  # type: ignore
        return config.task

    def _info(self) -> ds.DatasetInfo:
        processor: MsCocoProcessor = self.config.processor  # type: ignore
        features = processor.get_features(decode_rle=self.config.decode_rle)  # type: ignore
        return ds.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            features=features,
        )

    def _split_generators(self, dl_manager: ds.DownloadManager):
        file_paths = dl_manager.download_and_extract(_URLS[f"{self.year}"])

        imgs = file_paths["images"]  # type: ignore
        anns = file_paths["annotations"]  # type: ignore

        return [
            # ds.SplitGenerator(
            #     name=ds.Split.TRAIN,  # type: ignore
            #     gen_kwargs={
            #         "base_image_dir": imgs["train"],
            #         "base_annotation_dir": anns["train_validation"],
            #         "split": "train",
            #     },
            # ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,  # type: ignore
                gen_kwargs={
                    "base_image_dir": imgs["validation"],
                    "base_annotation_dir": anns["train_validation"],
                    "split": "val",
                },
            ),
            # ds.SplitGenerator(
            #     name=ds.Split.TEST,  # type: ignore
            #     gen_kwargs={
            #         "base_image_dir": imgs["test"],
            #         "test_image_info_path": anns["test_image_info"],
            #         "split": "test",
            #     },
            # ),
        ]

    def _generate_train_val_examples(
        self, split: str, base_image_dir: str, base_annotation_dir: str
    ):
        image_dir = os.path.join(base_image_dir, f"{split}{self.year}")

        ann_dir = os.path.join(base_annotation_dir, "annotations")
        ann_file_path = os.path.join(ann_dir, f"{self.task}_{split}{self.year}.json")

        processor: MsCocoProcessor = self.config.processor  # type: ignore

        ann_json = processor.load_annotation_json(ann_file_path=ann_file_path)

        # info = AnnotationInfo.from_dict(ann_json["info"])
        licenses = processor.load_licenses_data(license_dicts=ann_json["licenses"])
        images = processor.load_images_data(image_dicts=ann_json["images"])

        category_dicts = ann_json.get("categories")
        categories = (
            processor.load_categories_data(category_dicts=category_dicts)
            if category_dicts is not None
            else None
        )

        config: MsCocoConfig = self.config  # type: ignore
        yield from processor.generate_examples(
            annotations=processor.load_data(
                ann_dicts=ann_json["annotations"],
                images=images,
                decode_rle=config.decode_rle,
            ),
            categories=categories,
            image_dir=image_dir,
            images=images,
            licenses=licenses,
        )

    def _generate_test_examples(self, test_image_info_path: str):
        raise NotImplementedError

    def _generate_examples(
        self,
        split: MscocoSplits,
        base_image_dir: Optional[str] = None,
        base_annotation_dir: Optional[str] = None,
        test_image_info_path: Optional[str] = None,
    ):
        if split == "test" and test_image_info_path is not None:
            yield from self._generate_test_examples(
                test_image_info_path=test_image_info_path
            )
        elif (
            split in get_args(MscocoSplits)
            and base_image_dir is not None
            and base_annotation_dir is not None
        ):
            yield from self._generate_train_val_examples(
                split=split,
                base_image_dir=base_image_dir,
                base_annotation_dir=base_annotation_dir,
            )
        else:
            raise ValueError(
                f"Invalid arguments: split = {split}, "
                f"base_image_dir = {base_image_dir}, "
                f"base_annotation_dir = {base_annotation_dir}, "
                f"test_image_info_path = {test_image_info_path}",
            )
