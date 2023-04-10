# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Huggingface predict file for Image classification mlflow model."""

import base64
import io
import logging

import numpy as np
import pandas as pd
import tempfile

import re
import requests
import torch

from datasets import load_dataset
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
    TrainingArguments,
    Trainer,
)
from typing import List, Dict, Any, Optional, Tuple

logger = logging.getLogger(__name__)


class HFTaskLiterals:
    """HF task name constants."""

    MULTI_CLASS_IMAGE_CLASSIFICATION = "image-classification"
    MULTI_LABEL_CLASSIFICATION = "image-classification-multilabel"


class HFMiscellaneousConstants:
    """HF miscellaneous constants."""

    DEFAULT_IMAGE_KEY = "image"
    DEFAULT_THRESHOLD = 0.5


class MLFlowSchemaLiterals:
    """MLFlow model signature related schema."""

    INPUT_COLUMN_IMAGE = "image"
    OUTPUT_COLUMN_PROBS = "probs"
    OUTPUT_COLUMN_LABELS = "labels"


class MLFlowLiterals:
    """Constants related to MLFlow."""

    PROBS = "probs"
    LABELS = "labels"

    BATCH_SIZE_KEY = "batch_size"
    TRAIN_LABEL_LIST = "train_label_list"
    THRESHOLD = "threshold"


def create_temp_file(request_body: bytes, parent_dir: str) -> str:
    """Create temporory file, save image and return path to the file.

    :param request_body: Image
    :type request_body: bytes
    :param parent_dir: directory path
    :type parent_dir: str
    :return: Path to the file
    :rtype: str
    """
    with tempfile.NamedTemporaryFile(dir=parent_dir, mode="wb", delete=False) as image_file_fp:
        img_path = image_file_fp.name + ".png"
        img = Image.open(io.BytesIO(request_body))
        img.save(img_path)
        return img_path


def _process_image(img: pd.Series) -> pd.Series:
    """Process input image.

    If input image is in base64 string format, decode it to bytes.
    If input image is in url format, download it and return bytes.
    https://github.com/mlflow/mlflow/blob/master/examples/flower_classifier/image_pyfunc.py

    :param img: pandas series with image in base64 string format or url.
    :type img: pd.Series
    :return: decoded image in pandas series format.
    :rtype: Pandas Series
    """
    image = img[0]
    if _is_valid_url(image):
        image = requests.get(image).content
        return pd.Series(image)
    try:
        return pd.Series(base64.b64decode(image))
    except Exception:
        raise ValueError("Invalid image format")


def _is_valid_url(text: str) -> bool:
    """Check if text is url or base64 string.

    :param text: text to validate
    :type text: str
    :return: True if url else false
    :rtype: bool
    """
    regex = (
        "((http|https)://)(www.)?"
        + "[a-zA-Z0-9@:%._\\+~#?&//=]"
        + "{2,256}\\.[a-z]"
        + "{2,6}\\b([-a-zA-Z0-9@:%"
        + "._\\+~#?&//=]*)"
    )
    p = re.compile(regex)

    # If the string is empty
    # return false
    if str is None:
        return False

    # Return if the string
    # matched the ReGex
    if re.search(p, text):
        return True
    else:
        return False


def predict(input_data: pd.DataFrame, task, model, tokenizer, **kwargs) -> pd.DataFrame:
    """Perform inference on the input data.

    :param input_data: Input images for prediction.
    :type input_data: Pandas DataFrame with a first column name ['image'] of images where each
    image is in base64 String format.
    :param task: Task type of the model.
    :type task: HFTaskLiterals
    :param tokenizer: Preprocessing configuration loader.
    :type tokenizer: transformers.AutoImageProcessor
    :param model: Pytorch model weights.
    :type model: transformers.AutoModelForImageClassification
    :return: Output of inferencing
    :rtype: Pandas DataFrame with columns ['filename', 'probs', 'labels'] for classification and
    ['filename', 'boxes'] for object detection, instance segmentation
    """
    # Decode the base64 image column
    decoded_images = input_data.loc[:, [MLFlowSchemaLiterals.INPUT_COLUMN_IMAGE]].apply(axis=1, func=_process_image)

    # arguments for Trainer
    test_args = TrainingArguments(
        output_dir=".",
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=kwargs.get(MLFlowLiterals.BATCH_SIZE_KEY, 1),
        dataloader_drop_last=False,
        remove_unused_columns=False,
    )

    # To Do: change image height and width based on kwargs.

    with tempfile.TemporaryDirectory() as tmp_output_dir:
        image_path_list = decoded_images.iloc[:, 0].map(lambda row: create_temp_file(row, tmp_output_dir)).tolist()
        predicted_indexes, conf_scores = run_inference_batch(
            test_args,
            image_processor=tokenizer,
            model=model,
            image_path_list=image_path_list,
            task_type=task,
            threshold=kwargs.get(MLFlowLiterals.THRESHOLD, None),
        )

    df_result = pd.DataFrame(
        columns=[
            MLFlowSchemaLiterals.OUTPUT_COLUMN_PROBS,
            MLFlowSchemaLiterals.OUTPUT_COLUMN_LABELS,
        ]
    )

    labels = kwargs.get(MLFlowLiterals.TRAIN_LABEL_LIST)

    if task == HFTaskLiterals.MULTI_CLASS_IMAGE_CLASSIFICATION:
        predicted_labels = [labels[index] for index in predicted_indexes]
    else:
        predicted_labels = []
        for predicted_index in predicted_indexes:
            image_labels = []
            for index, pred in enumerate(predicted_index):
                if pred == 1:
                    image_labels.append(labels[index])
            predicted_labels.append(image_labels)

    df_result[MLFlowLiterals.PROBS], df_result[MLFlowLiterals.LABELS] = (
        conf_scores.tolist(),
        predicted_labels,
    )
    return df_result


def run_inference_batch(
    test_args: TrainingArguments,
    image_processor: AutoImageProcessor,
    model: AutoModelForImageClassification,
    image_path_list: List,
    task_type: HFTaskLiterals,
    threshold: Optional[float] = None,
) -> Tuple[torch.tensor, torch.tensor]:
    """Perform inference on batch of input images.

    :param test_args: Training arguments path.
    :type test_args: transformers.TrainingArguments
    :param image_processor: Preprocessing configuration loader.
    :type image_processor: transformers.AutoImageProcessor
    :param model: Pytorch model weights.
    :type model: transformers.AutoModelForImageClassification
    :param image_path_list: list of image paths for inferencing.
    :type image_path_list: List
    :param task_type: Task type of the model.
    :type task_type: HFTaskLiterals
    :param threshold: threshold for multi_label_classification
    :type threshold: optional, float
    :return: Predicted labels index, Predicted probabilities
    :rtype: Tuple of torch.tensor, torch.tensor
    """

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.tensor]:
        """Collator function for eval dataset.

        :param examples: List of input images.
        :type examples: List
        :return: Dictionary of pixel values in torch tensor format.
        :rtype: Dict
        """
        images = [data[HFMiscellaneousConstants.DEFAULT_IMAGE_KEY] for data in examples]
        return image_processor(images, return_tensors="pt")

    inference_dataset = load_dataset("imagefolder", data_files={"val": image_path_list})
    inference_dataset = inference_dataset["val"]

    # Initialize the trainer
    trainer = Trainer(
        model=model,
        args=test_args,
        tokenizer=image_processor,
        data_collator=collate_fn,
    )
    results = trainer.predict(inference_dataset)
    if task_type == HFTaskLiterals.MULTI_CLASS_IMAGE_CLASSIFICATION:
        probs = torch.nn.functional.softmax(torch.from_numpy(results.predictions), dim=1).numpy()
        y_pred = np.argmax(probs, axis=1)
    elif task_type == HFTaskLiterals.MULTI_LABEL_CLASSIFICATION:
        sigmoid = torch.nn.Sigmoid()
        threshold = threshold or HFMiscellaneousConstants.DEFAULT_THRESHOLD

        probs = sigmoid(torch.from_numpy(results.predictions)).numpy()
        y_pred = np.zeros(probs.shape)
        # next, use threshold to turn them into integer predictions
        y_pred[np.where(probs >= threshold)] = 1

    return y_pred, probs
