# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Huggingface predict file for Image classification MLflow model."""

import io
import logging

import pandas as pd
import torch

from typing import Any, Dict, Generator, List

from datasets import Dataset
from PIL import Image
from transformers import TrainingArguments, Trainer

try:
    # Use try/except since vision_utils is added as part of model export and not available when initializing
    # model wrapper for save_model().
    from vision_utils import process_image
except ImportError:
    pass

logger = logging.getLogger(__name__)


class HFTaskLiterals:
    """HF task name constants."""

    MULTI_CLASS_IMAGE_CLASSIFICATION = "image-classification"
    MULTI_LABEL_CLASSIFICATION = "image-classification-multilabel"


class HFMiscellaneousConstants:
    """HF miscellaneous constants."""

    DEFAULT_IMAGE_KEY = "image"
    DEFAULT_THRESHOLD = 0.5


class MLflowSchemaLiterals:
    """MLflow model signature related schema."""

    INPUT_COLUMN_IMAGE = "image"
    OUTPUT_COLUMN_PROBS = "probs"
    OUTPUT_COLUMN_LABELS = "labels"


class MLflowLiterals:
    """Constants related to MLflow."""

    PROBS = "probs"
    LABELS = "labels"

    BATCH_SIZE_KEY = "batch_size"
    TRAIN_LABEL_LIST = "train_label_list"
    THRESHOLD = "threshold"


def predict(input_data: pd.DataFrame, task, model, tokenizer, **kwargs) -> pd.DataFrame:
    """Perform inference on the input data.

    :param input_data: Input images for prediction.
    :type input_data: Pandas DataFrame with a first column name ['image'] of images where each
    image is in base64 String format.
    :param task: Task type of the model.
    :type task: HFTaskLiterals
    :param tokenizer: Image preprocessing object.
    :type tokenizer: transformers.AutoImageProcessor
    :param model: Pytorch model weights.
    :type model: transformers.AutoModelForImageClassification
    :return: Output of inferencing
    :rtype: Pandas DataFrame with columns ['filename', 'probs', 'labels'] for classification and
    ['filename', 'boxes'] for object detection, instance segmentation
    """
    # Make generator that reads all the input images and collator that produces B*C*H*W tensors.
    def image_generator_fn() -> Generator[Dict[str, Any], None, None]:
        for image in input_data[HFMiscellaneousConstants.DEFAULT_IMAGE_KEY]:
            yield {"image": Image.open(io.BytesIO(process_image(image)))}

    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.tensor]:
        """Collator function for eval dataset.

        :param examples: List of input images.
        :type examples: List
        :return: Dictionary of pixel values in torch tensor format.
        :rtype: Dict
        """
        images = [data["image"] for data in examples]
        return tokenizer(images, return_tensors="pt")
    
    # TODO: use in-memory dataset throughout the code (eg instead of list of images).

    # TODO: change image height and width based on kwargs.
    # Do inference (output: logits).
    inference_dataset = Dataset.from_generator(image_generator_fn)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=".",
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=kwargs.get(MLflowLiterals.BATCH_SIZE_KEY, 1),
            dataloader_drop_last=False,
            remove_unused_columns=False,
        ),
        data_collator=collate_fn,
    )
    results = trainer.predict(inference_dataset)

    # Calculate probabilities from logits.
    if task == HFTaskLiterals.MULTI_CLASS_IMAGE_CLASSIFICATION:
        probs = torch.nn.functional.softmax(torch.from_numpy(results.predictions), dim=1).numpy()
    elif task == HFTaskLiterals.MULTI_LABEL_CLASSIFICATION:
        sigmoid = torch.nn.Sigmoid()
        probs = sigmoid(torch.from_numpy(results.predictions)).numpy()

    # Convert to Pandas dataframe and return.
    df_result = pd.DataFrame(
        columns=[
            MLflowSchemaLiterals.OUTPUT_COLUMN_PROBS,
            MLflowSchemaLiterals.OUTPUT_COLUMN_LABELS,
        ]
    )
    df_result[MLflowLiterals.PROBS] = probs.tolist()
    df_result[MLflowLiterals.LABELS] = [kwargs.get(MLflowLiterals.TRAIN_LABEL_LIST).tolist()] * len(probs)

    return df_result
