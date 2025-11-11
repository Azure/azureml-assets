# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Azure AI Content Safety (AACS) handler module.

This module provides content safety validation using Azure AI Content Safety service.
It analyzes text and image content for harmful content categories including hate, self-harm,
sexual content, and violence.
"""
import os
import numpy as np
import pandas as pd
import base64
import io
import re
from PIL import Image
from typing import Dict, Union

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import (AnalyzeTextOptions,
                                           AnalyzeImageOptions,
                                           ImageData,
                                           AnalyzeTextResult,
                                           TextCategory)
from azure.core.credentials import AzureKeyCredential
from azure.core.pipeline.policies import HeadersPolicy
from azure.identity import ManagedIdentityCredential
from azure.mgmt.cognitiveservices import CognitiveServicesManagementClient
from mlflow.pyfunc.scoring_server import _get_jsonable_obj

from foundation.model.serve.api_server_setup.protocol import CompletionResponse, ChatCompletionResponse
from foundation.model.serve.logging_config import configure_logger
from foundation.model.serve.constants import EnvironmentVariables, CommonConstants

logger = configure_logger(__name__)


class AACSValidator:
    """Azure AI Content Safety validator for analyzing content safety.
    
    This class provides functionality to validate text and image content
    using Azure AI Content Safety service. It checks content against
    severity thresholds for harmful content categories.
    """

    def __init__(self):
        """Initialize the AACSValidator with content safety client and threshold settings."""
        self.g_aacs_client = None
        self.g_aacs_threshold = int(os.environ.get(
            EnvironmentVariables.CONTENT_SAFETY_THRESHOLD, CommonConstants.CONTENT_SAFETY_THERESHOLD_DEFAULT))
        self.aacs_setup()

    def get_aacs_access_key(self):
        """Get Azure AI Content Safety access key using managed identity.
        
        Returns:
            str: The access key for the AACS account.
            
        Raises:
            RuntimeError: If UAI_CLIENT_ID is not set.
        """
        uai_client_id = os.environ.get(EnvironmentVariables.UAI_CLIENT_ID)
        subscription_id = os.environ.get(EnvironmentVariables.SUBSCRIPTION_ID)
        resource_group_name = os.environ.get(
            EnvironmentVariables.RESOURCE_GROUP_NAME)
        aacs_account_name = os.environ.get(
            EnvironmentVariables.CONTENT_SAFETY_ACCOUNT_NAME)

        if not uai_client_id:
            raise RuntimeError(
                "Cannot get AACS access key, UAI_CLIENT_ID is not set, exiting...",
            )
        credential = ManagedIdentityCredential(client_id=uai_client_id)
        cs_client = CognitiveServicesManagementClient(
            credential, subscription_id)
        key = cs_client.accounts.list_keys(
            resource_group_name=resource_group_name,
            account_name=aacs_account_name,
        ).key1

        return key

    def aacs_setup(self):
        """Create an AACS endpoint for the server to check input and outputs.
        
        Returns:
            Exception or None: Exception if setup fails, None otherwise.
        """
        AACS_error = None
        try:
            endpoint = os.environ.get(
                EnvironmentVariables.CONTENT_SAFETY_ENDPOINT, None)
            key = self.get_aacs_access_key()

            if not endpoint:
                raise Exception(
                    "CONTENT_SAFETY_ENDPOINT env not set for AACS.")
            if not key:
                raise Exception("CONTENT_SAFETY_KEY env not set for AACS.")

            # Create a Content Safety client
            headers_policy = HeadersPolicy()
            headers_policy.add_header(
                "ms-azure-ai-sender", "fm-optimized-inference")
            self.g_aacs_client = ContentSafetyClient(
                endpoint,
                AzureKeyCredential(key),
                headers_policy=headers_policy,
            )
            logger.info("AACS client created successfully.")
        except Exception as e:
            logger.info(f"Error occurred while setting up aacs {e}")
            AACS_error = e

        return AACS_error

    def iterate(self, obj, current_key=None):
        """Iterate through object and check content severity recursively.
        
        Args:
            obj: The object to analyze (dict, list, DataFrame, or string).
            current_key: The current key being processed (optional).
            
        Returns:
            tuple: A tuple containing the sanitized object and maximum severity found.
        """
        if isinstance(obj, dict):
            severity = 0
            for key, value in obj.items():
                obj[key], value_severity = self.iterate(value, current_key=key)
                severity = max(severity, value_severity)
            return obj, severity
        elif isinstance(obj, list) or isinstance(obj, np.ndarray):
            severity = 0
            for idx in range(len(obj)):
                obj[idx], value_severity = self.iterate(
                    obj[idx], current_key=current_key)
                severity = max(severity, value_severity)
            return obj, severity
        elif isinstance(obj, pd.DataFrame):
            severity = 0
            columns = list(obj.columns)
            for i in range(obj.shape[0]):  # iterate over rows
                for j in range(obj.shape[1]):  # iterate over columns
                    obj.at[i, j], value_severity = self.iterate(
                        obj.at[i, j], current_key=columns[j])
                    severity = max(severity, value_severity)
            return obj, severity
        elif isinstance(obj, str):
            if self.is_valid_base64_image(obj):
                severity = self.analyze_image(obj)
            else:
                severity = self.analyze_text(obj)
            if severity > self.g_aacs_threshold:
                return "", severity
            else:
                return obj, severity
        else:
            return obj, 0

    def get_safe_response(self, result: Union[Dict, CompletionResponse, ChatCompletionResponse]):
        """Check if response is safe and sanitize if necessary.
        
        Args:
            result: The response to validate (dict or response object).
            
        Returns:
            The sanitized response with unsafe content removed.
        """
        logger.info("Analyzing response...")
        jsonable_result = _get_jsonable_obj(result, pandas_orient="records")
        if not self.g_aacs_client:
            return jsonable_result

        result, severity = self.iterate(jsonable_result)
        logger.info(f"Response analyzed, severity {severity}")
        return result

    def get_safe_input(self, input_data: Dict):
        """Check if input is safe and sanitize if necessary.
        
        Args:
            input_data: The input data to validate.
            
        Returns:
            tuple: A tuple containing the sanitized input and severity level.
        """
        if not self.g_aacs_client:
            return input_data, 0
        logger.info("Analyzing input...")
        result, severity = self.iterate(input_data)
        logger.info(f"Input analyzed, severity {severity}")
        return result, severity

    def analyze_response(self, aacs_response: AnalyzeTextResult):
        """Analyze AACS response and extract maximum severity.
        
        Args:
            aacs_response: The AACS analysis result.
            
        Returns:
            int: The maximum severity level found across all categories.
        """
        severity = 0
        aacs_category_set = [TextCategory.HATE, TextCategory.SELF_HARM,
                             TextCategory.SEXUAL, TextCategory.VIOLENCE]
        if aacs_response.categories_analysis:
            for category in aacs_response.categories_analysis:
                if category.category in aacs_category_set and category.severity > 0:
                    logger.info(
                        f"Analyzing aacs response for category {category.category} with severity {category.severity}")
                    severity = max(severity, category.severity)
        return severity

    def analyze_text(self, text: str):
        """Analyze text content for safety violations.
        
        Args:
            text: The text to analyze.
            
        Returns:
            int: The maximum severity level found in the text.
        """
        # Chunk text
        logger.info("Analyzing ...")
        if (not text) or (not text.strip()):
            return 0
        chunking_utils = CsChunkingUtils(chunking_n=1000, delimiter=".")
        split_text = chunking_utils.split_by(text)

        result = [self.analyze_response(self.g_aacs_client.analyze_text(
            AnalyzeTextOptions(text=i))) for i in split_text]
        severity = max(result)
        logger.info(f"Analyzed, severity {severity}")

        return severity

    def analyze_image(self, image_in_byte64: str) -> int:
        """Analyze image severity using azure content safety service.

        :param image_in_byte64: image in base64 format
        :type image_in_byte64: str
        :return: maximum severity of all categories
        :rtype: int
        """
        print("Analyzing image...")
        if image_in_byte64 is None:
            return 0
        request = AnalyzeImageOptions(image=ImageData(content=image_in_byte64))
        safety_response = self.g_aacs_client.analyze_image(request)
        severity = self.analyze_response(safety_response)
        print(f"Image Analyzed, severity {severity}")
        return severity

    def is_valid_base64_image(self, base64_string):
        """
        Check if a string is a valid base64-encoded image.

        Args:
            base64_string (str): The string to validate.

        Returns:
            bool: True if the string is a valid base64-encoded image, False otherwise.
        """
        try:
            # Remove data URL prefix if present (e.g., "data:image/png;base64,")
            if base64_string.startswith('data:'):
                # Extract base64 part after comma
                if ',' in base64_string:
                    base64_string = base64_string.split(',', 1)[1]
                else:
                    return False

            # Remove any whitespace
            base64_string = base64_string.strip()

            # Check if string contains only valid base64 characters
            base64_pattern = re.compile(r'^[A-Za-z0-9+/]*={0,2}$')
            if not base64_pattern.match(base64_string):
                return False

            # Check if length is valid for base64 (must be multiple of 4)
            if len(base64_string) % 4 != 0:
                return False

            # Try to decode base64
            try:
                decoded_data = base64.b64decode(base64_string, validate=True)
            except Exception as e:
                logger.info(f"Base64 decoding error: {e}")
                return False

            # Try to open as image using PIL
            try:
                image = Image.open(io.BytesIO(decoded_data))
                image_format = image.format.lower() if image.format else None

                # Verify it's a supported image format
                supported_formats = ['jpeg', 'jpg',
                                     'png', 'gif', 'bmp', 'webp', 'tiff']
                if image_format not in supported_formats:
                    return False

                # Additional validation - try to load the image data
                image.load()

                return True

            except Exception as e:
                logger.info(f"PIL image validation error: {e}")
                return False

        except Exception as e:
            logger.info(f"Unexpected error during base64 image validation: {e}")
            return False


class CsChunkingUtils:
    """Content safety chunking utilities for splitting text into analyzable chunks."""

    def __init__(self, chunking_n=1000, delimiter="."):
        """Initialize the chunking utility.
        
        Args:
            chunking_n: Maximum chunk size (default: 1000).
            delimiter: String delimiter for splitting (default: ".").
        """
        self.delimiter = delimiter
        self.chunking_n = chunking_n

    def chunkstring(self, string, length):
        """Chunk string into segments of a given length.
        
        Args:
            string: The string to chunk.
            length: Maximum length of each chunk.
            
        Yields:
            str: Chunks of the input string.
        """
        return (string[0 + i: length + i] for i in range(0, len(string), length))

    def split_by(self, input):
        """Split the input text intelligently by delimiter while respecting chunk size.
        
        Args:
            input: The input text to split.
            
        Returns:
            list: List of text chunks.
        """
        max_n = self.chunking_n
        split = [e + self.delimiter for e in input.split(self.delimiter) if e]
        ret = []
        buffer = ""

        for i in split:
            if len(i) > max_n:
                ret.append(buffer)
                ret.extend(list(self.chunkstring(i, max_n)))
                buffer = ""
                continue
            if len(buffer) + len(i) <= max_n:
                buffer = buffer + i
            else:
                ret.append(buffer)
                buffer = i

        if len(buffer) > 0:
            ret.append(buffer)
        return ret
