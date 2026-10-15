# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test validate_assets script."""

from pathlib import Path
from importlib import import_module
from unittest.mock import Mock
import pytest
import re
import shutil

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


@pytest.fixture
def model_validation_assets(tmp_path, monkeypatch):
    """Create two model assets with isolated model-specific validation checks."""
    source = Path(__file__).parent / RESOURCES_DIR / "azure-model"
    model_paths = {}
    for name in ("model-a", "model-b"):
        directory = tmp_path / name
        shutil.copytree(source, directory)
        for filename in ("asset.yaml", "spec.yaml"):
            path = directory / filename
            path.write_text(path.read_text(encoding="utf-8").replace("Azure-thing", name), encoding="utf-8")
        model_paths[name] = directory / assets.DEFAULT_ASSET_FILENAME

    module = import_module("azureml.assets.validate_assets")
    validate_model = Mock(return_value=0)
    confirm_results = Mock(return_value=0)
    monkeypatch.setattr(module, "validate_model_spec", validate_model)
    monkeypatch.setattr(module, "confirm_model_validation_results", confirm_results)
    monkeypatch.setattr(
        module, "get_validated_models_assets_map",
        Mock(return_value={name: object() for name in model_paths}),
    )
    return module, model_paths, validate_model, confirm_results


@pytest.mark.parametrize("changed_names", [
    None, [], ["model-a"], ["model-b"], ["model-a", "model-b"], ["unrelated.txt"],
])
def test_changed_model_validation_scope(tmp_path, model_validation_assets, changed_names):
    """Apply both model-specific checks only to selected assets, or all when unfiltered."""
    module, model_paths, validate_model, confirm_results = model_validation_assets
    changed_files = None if changed_names is None else [
        model_paths[name] if name in model_paths else tmp_path / name for name in changed_names
    ]

    assert module.validate_assets(tmp_path, assets.DEFAULT_ASSET_FILENAME, changed_files=changed_files)

    expected = set(model_paths) if not changed_names else set(changed_names) & set(model_paths)
    assert {call.args[0].name for call in validate_model.call_args_list} == expected
    assert {call.args[0].name for call in confirm_results.call_args_list} == expected
    assert validate_model.call_count == len(expected)
    assert confirm_results.call_count == len(expected)


@pytest.mark.parametrize("failed_check", ["spec", "results"])
@pytest.mark.parametrize("changed_name,expected", [(None, False), ("model-a", True), ("model-b", False)])
def test_changed_model_validation_errors(tmp_path, model_validation_assets, failed_check, changed_name, expected):
    """Ignore errors in unchanged models without suppressing selected-model failures."""
    module, model_paths, validate_model, confirm_results = model_validation_assets
    check = validate_model if failed_check == "spec" else confirm_results
    check.side_effect = lambda config, *_: int(config.name == "model-b")
    changed_files = None if changed_name is None else [model_paths[changed_name]]

    assert module.validate_assets(
        tmp_path, assets.DEFAULT_ASSET_FILENAME, changed_files=changed_files,
    ) is expected


def test_changed_model_filter_preserves_spec(tmp_path, model_validation_assets):
    """Do not strip and rewrite SDK-unsupported fields in an unchanged model spec."""
    module, model_paths, _, _ = model_validation_assets
    spec = model_paths["model-b"].with_name("spec.yaml")
    original = spec.read_bytes() + b"\nvariantInfo: {}\n"
    spec.write_bytes(original)

    assert module.validate_assets(
        tmp_path, assets.DEFAULT_ASSET_FILENAME, changed_files=[model_paths["model-a"]],
    )
    assert spec.read_bytes() == original


def test_changed_model_filter_preserves_duplicates(tmp_path, model_validation_assets):
    """Keep repository-wide duplicate-name checks even when only one model changed."""
    module, model_paths, _, _ = model_validation_assets
    for filename in ("asset.yaml", "spec.yaml"):
        path = model_paths["model-b"].with_name(filename)
        path.write_text(path.read_text(encoding="utf-8").replace("model-b", "model-a"), encoding="utf-8")

    assert not module.validate_assets(
        tmp_path, assets.DEFAULT_ASSET_FILENAME, changed_files=[model_paths["model-a"]],
    )
