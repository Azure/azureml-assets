# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the IO utilities."""

import pytest
import os

from src.shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
)


@pytest.mark.unit
class TestIOUtils:
    def test_save_spark_df_as_mltable_duplicate_case_sensitive_columns(self):
        """Test the save dataframe as mltable functionality."""
        spark = init_spark()
        production_df = spark.createDataFrame([(1, "c", "bob"), (2, "d", "BOB")], ["id", "age", "Id"])
        save_spark_df_as_mltable(production_df, "localData")
        assert os.path.exists("localData/") is True
