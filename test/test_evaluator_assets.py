# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test evaluator assets validation."""

from pathlib import Path
import pytest
import yaml
from jsonschema import Draft202012Validator


EVALUATORS_DIR = (
    Path(__file__).parent.parent / "assets" / "evaluators" / "builtin"
)


def get_all_evaluator_spec_files():
    """Get all spec.yaml files from evaluators/builtin directory.

    Returns:
        list: List of tuples containing (evaluator_name, spec_file_path)
    """
    spec_files = []
    if not EVALUATORS_DIR.exists():
        return spec_files

    for evaluator_dir in EVALUATORS_DIR.iterdir():
        if evaluator_dir.is_dir():
            spec_file = evaluator_dir / "spec.yaml"
            if spec_file.exists():
                spec_files.append((evaluator_dir.name, spec_file))

    return spec_files


@pytest.mark.parametrize(
    "evaluator_name,spec_file_path", get_all_evaluator_spec_files()
)
def test_validate_evaluators_schema(
    evaluator_name: str, spec_file_path: Path
):
    """Test evaluator spec.yaml files have valid schemas.

    Args:
        evaluator_name (str): Name of the evaluator
        spec_file_path (Path): Path to the spec.yaml file
    """
    # Load the spec.yaml file
    with open(spec_file_path, 'r') as f:
        spec_data = yaml.safe_load(f)

    assert spec_data is not None, (
        f"Failed to load spec.yaml for {evaluator_name}"
    )

    # Validate initParameterSchema
    if "initParameterSchema" in spec_data:
        init_schema = spec_data["initParameterSchema"]
        try:
            # Check if the schema itself is valid
            Draft202012Validator.check_schema(init_schema)
        except Exception as e:
            pytest.fail(
                f"initParameterSchema validation failed for "
                f"{evaluator_name}: {str(e)}"
            )

    # Validate dataMappingSchema
    if "dataMappingSchema" in spec_data:
        data_mapping_schema = spec_data["dataMappingSchema"]
        try:
            # Check if the schema itself is valid
            Draft202012Validator.check_schema(data_mapping_schema)
        except Exception as e:
            pytest.fail(
                f"dataMappingSchema validation failed for "
                f"{evaluator_name}: {str(e)}"
            )

    # Validate outputSchema
    if "outputSchema" in spec_data:
        output_schema = spec_data["outputSchema"]
        try:
            # Check if the schema itself is valid
            Draft202012Validator.check_schema(output_schema)
        except Exception as e:
            pytest.fail(
                f"outputSchema validation failed for "
                f"{evaluator_name}: {str(e)}"
            )

    # Ensure at least one of the schemas exists
    assert "initParameterSchema" in spec_data or \
        "dataMappingSchema" in spec_data, (
            f"spec.yaml for {evaluator_name} must contain at least "
            f"initParameterSchema or dataMappingSchema"
        )


def test_all_evaluators_have_spec_files():
    """Test that all evaluator directories contain spec.yaml files."""
    evaluator_dirs = [d for d in EVALUATORS_DIR.iterdir() if d.is_dir()]
    assert len(evaluator_dirs) > 0, "No evaluator directories found"

    missing_spec_files = []
    for evaluator_dir in evaluator_dirs:
        spec_file = evaluator_dir / "spec.yaml"
        if not spec_file.exists():
            missing_spec_files.append(evaluator_dir.name)

    assert len(missing_spec_files) == 0, (
        f"The following evaluators are missing spec.yaml files: "
        f"{', '.join(missing_spec_files)}"
    )
