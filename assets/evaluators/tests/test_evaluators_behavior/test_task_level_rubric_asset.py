# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline metadata contract for the service-backed task rubric."""

from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

pytestmark = pytest.mark.unittest
ASSET_DIR = Path(__file__).parents[2] / "builtin" / "task_level_rubric"
SPEC = yaml.safe_load((ASSET_DIR / "spec.yaml").read_text(encoding="utf-8"))


class TestTaskLevelRubricAsset:
    """Keep executable code private while preserving the public contract."""

    def test_metadata_only_service(self):
        """Publishing must not upload Python, prompts, or executable paths."""
        assert {path.name for path in ASSET_DIR.iterdir()} == {
            "asset.yaml",
            "spec.yaml",
        }
        assert SPEC["version"] == 2
        assert SPEC["evaluatorSubType"] == "service"
        assert SPEC["supportedEvaluationLevels"] == ["conversation"]
        assert not {"path", "code", "code_text", "blob_uri"} & SPEC.keys()
        assert SPEC["outputSchema"]["task_level_rubric"]["type"] == "boolean"

    def test_initialization_contract(self):
        """A customer deployment and optional weighted threshold remain."""
        schema = SPEC["initParameterSchema"]
        validator = Draft202012Validator(schema)
        assert not list(validator.iter_errors({"deployment_name": "judge"}))
        assert list(validator.iter_errors({}))
        for threshold in (-0.1, 1.1, True):
            assert list(
                validator.iter_errors(
                    {"deployment_name": "judge", "threshold": threshold}
                )
            )

    @pytest.mark.parametrize(
        "query,response",
        [
            ("Question", "Answer"),
            (
                [{"role": "user", "content": "Question"}],
                [{"role": "assistant", "content": "Answer"}],
            ),
        ],
    )
    def test_row_contract(self, query, response):
        """Structured row rubrics and both q/r forms remain supported."""
        row = {
            "query": query,
            "response": response,
            "rubric_dimensions": [
                {
                    "id": "quality",
                    "description": "Answer the question",
                    "weight": 1,
                    "always_applicable": False,
                }
            ],
        }
        validator = Draft202012Validator(SPEC["dataMappingSchema"])
        assert not list(validator.iter_errors(row))
        for field in ("query", "response", "rubric_dimensions"):
            assert list(
                validator.iter_errors(
                    {key: value for key, value in row.items() if key != field}
                )
            )
        assert list(validator.iter_errors({**row, "messages": None}))
        assert list(validator.iter_errors({**row, "rubric_dimensions": "[]"}))

    @pytest.mark.parametrize("weight", [0, -1, 1.5, True, "1"])
    def test_invalid_dimension_weight(self, weight):
        """Dimension weights stay positive integers, never booleans."""
        row = {
            "query": "Question",
            "response": "Answer",
            "rubric_dimensions": [
                {
                    "id": "quality",
                    "description": "Answer",
                    "weight": weight,
                }
            ],
        }
        validator = Draft202012Validator(SPEC["dataMappingSchema"])
        assert list(validator.iter_errors(row))
