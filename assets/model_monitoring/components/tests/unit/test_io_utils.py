# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the IO utilities."""

import pytest
import os

from src.shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
)
from tests.e2e.utils.io_utils import create_pyspark_dataframe


@pytest.mark.unit
class TestDFUtils:
    def test_save_spark_df_as_mltable(self):
        """Test the save dataframe as mltable functionality."""
        spark = init_spark()
        production_df = spark.createDataFrame([(1, "c"), (2, "d")], ["id", "age", "Id"])
        save_spark_df_as_mltable(production_df, "localData")
        assert os.path.exists("localData/") == True