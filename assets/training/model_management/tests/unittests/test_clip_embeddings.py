# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test CLIP Embeddings helper code."""
import base64
import pytest
import pandas as pd

# add clip directory to sys path to resolve imports
import os
import sys
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "./azureml/model/mgmt/processors/pyfunc/clip"))
sys.path.append(MODEL_DIR)
from azureml.model.mgmt.processors.pyfunc.clip.clip_embeddings_mlflow_wrapper import (  # noqa: E402
    CLIPEmbeddingsMLFlowModelWrapper
)


class TestCLIPEmbeddings:
    """Test class for clip embeddings helper functions."""

    def test_valid_input_clip(self):
        """Test that the validate_input function for CLIP Embeddings model allows invalid input."""
        input_cases = InputCases()

        has_image, has_text = CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.valid_text_input())
        assert not has_image
        assert has_text

        has_image, has_text = CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.valid_text_input_nan())
        assert not has_image
        assert has_text

        has_image, has_text = CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.valid_image_input())
        assert has_image
        assert not has_text

        has_image, has_text = CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.valid_image_input_nan())
        assert has_image
        assert not has_text

        has_image, has_text = CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.valid_combination_input())
        assert has_image
        assert has_text

    def test_invalid_input_clip(self):
        """Test that the validate_input function for CLIP Embeddings model catches valid input."""
        input_cases = InputCases()

        with pytest.raises(ValueError):
            CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.invalid_text_input())

        with pytest.raises(ValueError):
            CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.invalid_image_input())

        with pytest.raises(ValueError):
            CLIPEmbeddingsMLFlowModelWrapper.validate_input(input_cases.invalid_combination_input())


class InputCases:
    """Class for generating valid and invalid inputs for CLIP Embeddings."""

    def __init__(self):
        """Initialize text and image data."""
        self.text_list = [
            "text 1", "text 2", "text 3"
        ]
        self.image_list = [
            base64.encodebytes(b"image with things").decode("utf-8"),
            base64.encodebytes(b"image with stuff").decode("utf-8"),
            base64.encodebytes(b"blank image").decode("utf-8"),
        ]

    def valid_text_input(self):
        """Return dataframe where all 'image' column is empty string and 'text' column has values."""
        text_data = [["", text] for text in self.text_list]
        test_df_text = pd.DataFrame(
            data=text_data,
            columns=["image", "text"],
        )
        return test_df_text

    def valid_image_input(self):
        """Return dataframe where all 'text' column is empty string and 'image' column has values."""
        image_data = [[img, ""] for img in self.image_list]
        test_df_image = pd.DataFrame(
            data=image_data,
            columns=["image", "text"],
        )
        return test_df_image

    def valid_combination_input(self):
        """Return dataframe where all 'image' column and 'text' column have values."""
        combine_data = [[self.image_list[i], self.text_list[i]] for i in range(0, len(self.image_list))]
        test_df_combine = pd.DataFrame(
            data=combine_data,
            columns=["image", "text"],
        )
        return test_df_combine

    def valid_text_input_nan(self):
        """Return dataframe where all 'image' column is NaN and 'text' column has values."""
        test_df_text = pd.DataFrame(
            columns=["image"],
        )
        test_df_text['text'] = self.text_list
        return test_df_text

    def valid_image_input_nan(self):
        """Return dataframe where all 'text' column is NaN and 'image' column has values."""
        test_df_image = pd.DataFrame(
            columns=["text"],
        )
        test_df_image['image'] = self.image_list
        return test_df_image

    def invalid_text_input(self):
        """Return dataframe where all 'image' column is empty string and 'text' column has missing values."""
        test_df = self.valid_text_input()
        test_df['text'].iloc[1] = ""
        return test_df

    def invalid_image_input(self):
        """Return dataframe where all 'text' column is empty string and 'image' column has missing values."""
        test_df = self.valid_image_input()
        test_df['image'].iloc[1] = ""
        return test_df

    def invalid_combination_input(self):
        """Return dataframe where 'text' column and 'image' column have missing values."""
        test_df = self.valid_combination_input()
        test_df["image"].iloc[1] = ""
        test_df["text"].iloc[1] = ""
        return test_df
