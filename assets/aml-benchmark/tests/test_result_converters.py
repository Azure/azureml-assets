# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the functionality of the prompt factory which powers the prompt crafter."""

import pytest
import sys

from .test_utils import get_src_dir


sys.path.append(get_src_dir())


from aml_benchmark.batch_output_formatter.result_converters import ResultConverters  # noqa: E402


class TestResultConverters:
    """Test result converters."""

    @pytest.mark.parametrize(
            'model_type,label_col', [('oai', 'label'), ('oss', None)]
    )
    def test_label_column_name(self, model_type, label_col):
        """Test label column name."""
        rc = ResultConverters(
            model_type, "metadata_key", "data_id_key", label_col, None, "fallback_value")
        expect_label = label_col if label_col else "ground_truth"
        assert rc.ground_truth_column_name == expect_label
