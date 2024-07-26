# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""This file contains unit tests for the IO utilities."""

import pytest
import os
import shutil

from src.shared_utilities.io_utils import (
    init_spark,
    save_spark_df_as_mltable,
)

LOCAL_DATA_FOLDER = "localData"


@pytest.fixture(scope="function")
def cleanup_data_folder():
    # test will run and data will be populated. Afterwards, Cleanup directory
    yield
    shutil.rmtree(f"{LOCAL_DATA_FOLDER}/")


@pytest.mark.unit
class TestIOUtils:
    """Test class for IO utilities."""

    def test_save_spark_df_as_mltable_normal(self, cleanup_data_folder):
        """Test the save dataframe as mltable functionality."""
        spark = init_spark()
        production_df = spark.createDataFrame([(1, "c"), (2, "d")], ["id", "age"])
        save_spark_df_as_mltable(production_df, LOCAL_DATA_FOLDER)
        assert os.path.exists("{LOCAL_DATA_FOLDER}/") is True
        assert os.path.exists("{LOCAL_DATA_FOLDER}/data/") is True

        saved_df = spark.read.parquet("{LOCAL_DATA_FOLDER}/data")
        print("Saved dataframe:")
        saved_df.show()
        assert "id" in saved_df.columns
        assert "age" in saved_df.columns
