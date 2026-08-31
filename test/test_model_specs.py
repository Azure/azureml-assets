# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for model specifications."""

from pathlib import Path
import re


RETIRED_NCSV3_SKU_PATTERN = re.compile(r"Standard_NC(?:6|12|24|24r)s_v3")


def test_model_specs_do_not_reference_retired_ncsv3_skus():
    """Ensure model specs do not reference retired NCsv3 SKUs."""
    repo_root = Path(__file__).parents[1]
    affected_specs = [
        str(spec.relative_to(repo_root))
        for spec in repo_root.glob("assets/models/**/spec.yaml")
        if RETIRED_NCSV3_SKU_PATTERN.search(spec.read_text())
    ]

    assert affected_specs == []
