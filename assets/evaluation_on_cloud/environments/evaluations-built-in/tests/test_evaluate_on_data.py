# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for evaluate_on_data.py."""

import sys
from pathlib import Path
import os
import importlib


def test_evaluate_obo_sets_env_var():
    """Test that importing evaluate_on_data.py sets the AZUREML_OBO_ENABLED environment variable to "True"."""
    script_path = Path(__file__).parent.parent / "context" / "evaluate_on_data.py"
    module_name = "evaluate_on_data"
    sys.path.insert(0, str(script_path.parent))

    if module_name in sys.modules:
        del sys.modules[module_name]

    assert os.environ.get("AZUREML_OBO_ENABLED") is None

    importlib.import_module(module_name)

    assert os.environ.get("AZUREML_OBO_ENABLED") == "True"


def test_model_target_sets_credential(monkeypatch):
    """Test that ModelTarget sets the credential property correctly."""
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from context.model_target import ModelTarget

    # Case 1: api_key is provided
    mt = ModelTarget(
        endpoint="https://dummy.endpoint",
        api_key="dummy_api_key",
        model_params={},
        system_message="sys",
        few_shot_examples=[],
    )
    assert mt.credential == "dummy_api_key"

    # Case 2: api_key is None, should use AzureMLOnBehalfOfCredential
    class DummyCredential:
        def get_token(self, scope):
            class Token:
                token = "obo_token"

            return Token()

    monkeypatch.setattr("context.model_target.AzureMLOnBehalfOfCredential", DummyCredential)
    mt2 = ModelTarget(
        endpoint="https://dummy.endpoint",
        api_key=None,
        model_params={},
        system_message="sys",
        few_shot_examples=[],
    )
    assert mt2.credential == "obo_token"

