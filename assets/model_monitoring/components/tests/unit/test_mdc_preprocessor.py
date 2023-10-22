# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for mdc preprocessor."""

import pytest
import fsspec
import shutil
import os
import sys
import spark_mltable  # noqa, to enable spark.read.mltable
import pandas as pd
from pandas.testing import assert_frame_equal
from model_data_collector_preprocessor.run import (
    _raw_mdc_uri_folder_to_preprocessed_spark_df,
    mdc_preprocessor,
)


@pytest.mark.unit
class TestMDCPreprocessor:
    """Test class for MDC Preprocessor."""

    @pytest.mark.parametrize(
        "window_start_time, window_end_time, extract_correlation_id",
        [
            # data only
            ("2023-10-11T20:00:00", "2023-10-11T21:00:00", True),
            ("2023-10-11T20:00:00", "2023-10-11T21:00:00", False),
            # data and dataref mix
            ("2023-10-15T17:00:00", "2023-10-15T18:00:00", True),
            ("2023-10-15T17:00:00", "2023-10-15T18:00:00", False),
            # dataref only
            ("2023-10-16T21:00:00", "2023-10-16T22:00:00", True),
            ("2023-10-16T21:00:00", "2023-10-16T22:00:00", False),
        ]
    )
    def test_uri_folder_to_spark_df(self, window_start_time, window_end_time, extract_correlation_id):
        """Test uri_folder_to_spark_df()."""
        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())
        python_path = sys.executable
        os.environ["PYSPARK_PYTHON"] = python_path
        os.environ["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH','')};./src"
        fs = fsspec.filesystem("file")
        preprocessed_output = "assets/model_monitoring/components/tests/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp", True)
        sdf = _raw_mdc_uri_folder_to_preprocessed_spark_df(
            window_start_time,
            window_end_time,
            "assets/model_monitoring/components/tests/unit/raw_mdc_data/",
            preprocessed_output,
            extract_correlation_id,
            fs,
        )
        print("preprocessed dataframe:")
        sdf.show(truncate=False)
        pdf_actual = sdf.toPandas()

        pdf_expected = pd.DataFrame({
            'sepal_length': [1, 2, 3, 1],
            'sepal_width': [2.3, 3.2, 3.4, 1.0],
            'petal_length': [2, 3, 3, 4],
            'petal_width': [1.3, 1.5, 1.8, 1.6]
        })
        if extract_correlation_id:
            pdf_expected['correlationid'] = [
                '7f16d5b1-76f9-4b3e-b82d-fc21d29356a5_0',
                'f2b524a7-3272-45df-a530-c945004de305_0',
                'f2b524a7-3272-45df-a530-c945004de305_1',
                '95e1afa0-256d-414b-8e4c-fea1baa98225_0'
            ]
        pd.set_option('display.max_colwidth', -1)
        pd.set_option('display.max_columns', 10)
        print(pdf_expected)
        pd.reset_option('display.max_colwidth')
        pd.reset_option('display.max_columns')

        assert_frame_equal(pdf_actual, pdf_expected)

    @pytest.mark.skip(reason="spark write is not ready in local")
    def test_mdc_preprocessor(self):
        """Test mdc_preprocessor()."""
        print("testing test_mdc_preprocessor...")
        os.environ["PYSPARK_PYTHON"] = sys.executable
        fs = fsspec.filesystem("file")
        preprocessed_output = "tests/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp")
        mdc_preprocessor(
            "2023-10-11T20:00:00",
            "2023-10-11T21:00:00",
            "tests/unit/raw_mdc_data/",
            preprocessed_output,
            False,
            fs,
        )
