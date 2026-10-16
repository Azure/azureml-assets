# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test validate_assets script."""

from pathlib import Path
import pytest
import re

import azureml.assets as assets

RESOURCES_DIR = Path("resources/validate")
MODEL_VALIDATION_RESULTS = Path("resources/model_validation_results")


@pytest.mark.parametrize(
    "test_subdir,check_images,check_names,check_names_skip_pattern,expected",
    [
        ("name-mismatch", False, True, None, False),
        ("version-mismatch", False, True, None, False),
        ("invalid-strings", False, True, None, False),
        ("env-with-underscores", False, True, None, False),
        ("framework-ver-missing", False, True, None, False),
        ("ubuntu-in-name", False, True, None, False),
        ("ubuntu-in-name", False, True, re.compile(r"environment/env-ubuntu20.04/.+"), True),
        ("ubuntu-in-name", False, False, None, True),
        ("extra-gpu", False, None, True, False),
        ("incorrect-order", False, True, None, False),
        ("image-name-mismatch", True, True, None, False),
        ("publishing-disabled", True, True, None, False),
        ("good-validation", True, True, None, True),
        ("correct-order", True, True, None, True),
        ("missing-description-file", True, True, None, False),
        ("data-good", False, True, None, True),
        ("data-path-mismatch-1", False, True, None, False),
        ("data-path-mismatch-2", False, True, None, False),
        ("dockerfile-from-ce-image", False, False, None, False),
        ("dockerfile-from-ce-image-comment", False, False, None, False),
        ("dockerfile-from-ce-image-windows", False, False, None, False),
        ("bad-build-context", False, True, None, False),
        ("model-with-azure", False, True, None, False),
        ("azure-model", False, True, None, True),
        ("azure-model-bad", False, True, None, False),
        ("model-microsoft-good", False, True, None, True),
        ("model-microsoft-bad-asset-name", False, True, None, False),
        ("model-microsoft-bad-spec", False, True, None, False),
        ("evaluationresult/invalid_evaluation_type", False, True, None, False),
        ("evaluationresult/text_embeddings_correct", False, True, None, True),
        ("evaluationresult/text_embeddings_incorrect", False, True, None, False),
        ("evaluationresult/text_generation_correct", False, True, None, True),
        ("evaluationresult/text_generation_updated_task", False, True, None, True),
        ("evaluationresult/text_generation_incorrect", False, True, None, False),
        ("evaluationresult/vision_correct", False, True, None, True),
        ("evaluationresult/vision_incorrect", False, True, None, False),
        ("evaluationresult/text_cost_correct", False, True, None, True),
        ("evaluationresult/text_cost_incorrect", False, True, None, False),
        ("evaluationresult/text_quality_correct", False, True, None, True),
        ("evaluationresult/text_quality_incorrect", False, True, None, False),
        ("evaluationresult/text_performance_correct", False, True, None, True),
        ("evaluationresult/text_performance_incorrect", False, True, None, False),
    ]
)
def test_validate_assets(test_subdir: str, check_images: bool, check_names: bool,
                         check_names_skip_pattern: re.Pattern, expected: bool):
    """Test validate_assets function.

    Args:
        test_subdir (str): Test subdirectory
        check_images (bool): Check image build/publish info
        check_names (bool): Check name
        check_names_skip_pattern (re.Pattern): Skip pattern for name validation
        expected (bool): Success expected
    """
    this_dir = Path(__file__).parent

    assert assets.validate_assets(
        input_dirs=this_dir / RESOURCES_DIR / test_subdir,
        model_validation_results_dir=this_dir / MODEL_VALIDATION_RESULTS / test_subdir,
        asset_config_filename=assets.DEFAULT_ASSET_FILENAME,
        check_names=check_names,
        check_names_skip_pattern=check_names_skip_pattern,
        check_images=check_images,
        check_build_context=True) == expected
