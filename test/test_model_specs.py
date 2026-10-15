# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for model specifications."""

from pathlib import Path
import re

import pytest


RETIRED_NCSV3_SKU_PATTERN = re.compile(r"Standard_NC(?:6|12|24|24r)s_v3")
RETIRED_NVV3_SKU_PATTERN = re.compile(r"Standard_NV(?:12|24|48)s_v3")


@pytest.mark.parametrize("retired_sku_pattern", [RETIRED_NCSV3_SKU_PATTERN, RETIRED_NVV3_SKU_PATTERN])
def test_model_specs_do_not_reference_retired_skus(retired_sku_pattern):
    """Ensure model specs do not reference retired NCsv3 or NVv3 SKUs."""
    repo_root = Path(__file__).parents[1]
    affected_specs = [
        str(spec.relative_to(repo_root))
        for spec in repo_root.glob("assets/models/**/spec.yaml")
        if retired_sku_pattern.search(spec.read_text())
    ]

    assert affected_specs == []
