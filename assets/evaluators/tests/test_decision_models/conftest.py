# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Offline released-SDK checks opt in via AZUREML_COMPAT_SDK_ROOT.

The root must contain azureml-compat-sdk-1.17.0, azureml-compat-sdk-1.18.2,
azureml-compat-sdk-1.18.8, and azureml-compat-sdk-1.18.9 isolated installations.
Tests never install packages or assume global temp targets.
"""

import os
from pathlib import Path

import pytest


@pytest.fixture
def released_sdk_target():
    """Resolve an explicitly selected offline SDK target or explain why it is skipped."""
    def resolve(version):
        root = os.environ.get("AZUREML_COMPAT_SDK_ROOT")
        if not root:
            pytest.skip(
                "Released SDK checks are opt-in: set AZUREML_COMPAT_SDK_ROOT to the parent "
                f"of an isolated azureml-compat-sdk-{version} installation."
            )
        target = Path(root) / f"azureml-compat-sdk-{version}"
        if not (target / "azure" / "ai" / "evaluation" / "__init__.py").is_file():
            pytest.skip(
                f"SDK {version} isolated target is missing: {target}. "
                "Provision it offline or update AZUREML_COMPAT_SDK_ROOT."
            )
        return target.resolve()

    return resolve
