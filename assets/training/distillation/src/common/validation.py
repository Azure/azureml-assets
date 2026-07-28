# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component validation utils."""

from pathlib import Path
from typing import List, Optional

from azureml.acft.common_components.utils.error_handling.exceptions import (
    ACFTValidationException,
)
from azureml.acft.common_components.utils.error_handling.error_definitions import (
    ACFTUserError,
)
from azureml._common._error_definition.azureml_error import AzureMLError

from common.constants import SUPPORTED_FILE_FORMATS, MAX_BATCH_SIZE


def validate_file_paths_with_supported_formats(file_paths: List[Optional[str]]):
    """Check if the file path is in the list of supported formats."""
    for file_path in file_paths:
        if file_path:
            file_suffix = Path(file_path).suffix.lower()
            file_ext = file_suffix.split("?")[0]
        if file_ext and file_ext not in SUPPORTED_FILE_FORMATS:
            raise ACFTValidationException._with_error(
                AzureMLError.create(
                    ACFTUserError,
                    pii_safe_message=(
                        f"{file_path} is not in list of supported file formats. "
                        f"Supported file formats: {SUPPORTED_FILE_FORMATS}"
                    ),
                )
            )


def validate_file_exists(file_paths: List[Optional[str]]):
    """Check if the file paths exist."""
    for file_path in file_paths:
        if file_path:
            file = Path(file_path)
            if not file.exists():
                raise ACFTValidationException._with_error(
                    AzureMLError.create(
                        ACFTUserError,
                        pii_safe_message=(f"File {file_path} does not exist."),
                    )
                )


def validate_model_temperature(temperature: float):
    """Validate if model temperature is well within limits."""
    if temperature < 0 or temperature > 1:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid teacher_model_temperature. ",
                    f"Value should 0<=val<=1, but is {temperature}",
                ),
            )
        )


def validate_model_top_p(top_p: float):
    """Validate if model top_p is well within limits."""
    if top_p < 0 or top_p > 1:
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid teacher_model_top_p. ",
                    f"Value should be 0<=val<=1, but is {top_p}",
                ),
            )
        )


def validate_model_frequency_penalty(val: float):
    """Validate if model frequency penalty is well within limits."""
    if val and (val < 0 or val > 2):
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid teacher_model_frequency_penalty. ",
                    f"Value should be 0<=val<=2, but is {val}",
                ),
            )
        )


def validate_model_presence_penalty(val: float):
    """Validate if model presence penalty is well within limits."""
    if val and (val < 0 or val > 2):
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid teacher_model_presence_penalty. ",
                    f"Value should be 0<=val<=2, but is {val}",
                ),
            )
        )


def validate_request_batch_size(val: int):
    """Validate if requested batch size is well within limits."""
    if val and (val <= 0 or val > MAX_BATCH_SIZE):
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid request_batch_size. ",
                    f"Value should be 0<=val<={MAX_BATCH_SIZE}, but is {val}",
                ),
            )
        )


def validate_min_endpoint_success_ratio(val: int):
    """Validate if requested endpoint success ration is well within limits."""
    if val and (val < 0 or val > 1):
        raise ACFTValidationException._with_error(
            AzureMLError.create(
                ACFTUserError,
                pii_safe_message=(
                    "Invalid min_endpoint_success_ration. ",
                    f"Value sould be 0<=val<=1, but is {val}",
                ),
            )
        )
