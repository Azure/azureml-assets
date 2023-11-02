# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for mdc preprocessor."""

import pytest
from unittest.mock import Mock
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
    _convert_to_azureml_long_form,
    _get_datastore_from_input_path,
)


@pytest.fixture(scope="module")
def mdc_preprocessor_test_setup():
    """Change working directory to root of the assets/model_monitoring_components."""
    original_work_dir = os.getcwd()
    momo_work_dir = os.path.abspath(f"{os.path.dirname(__file__)}/../..")
    os.chdir(momo_work_dir)  # change working directory to root of the assets/model_monitoring_components
    yield
    os.chdir(original_work_dir)  # change working directory back to original


@pytest.mark.unit
class TestMDCPreprocessor:
    """Test class for MDC Preprocessor."""

    @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
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
    def test_uri_folder_to_spark_df(self, mdc_preprocessor_test_setup,
                                    window_start_time, window_end_time, extract_correlation_id):
        """Test uri_folder_to_spark_df()."""
        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())
        python_path = sys.executable
        os.environ["PYSPARK_PYTHON"] = python_path
        print("PYSPARK_PYTHON", os.environ.get("PYSPARK_PYTHON", "NA"))
        module_path = f"{os.getcwd()}/src"
        old_python_path = os.environ.get("PYTHONPATH", None)
        old_python_path = f"{old_python_path};" if old_python_path else ""
        os.environ["PYTHONPATH"] = f"{old_python_path}{module_path}"
        print("PYTHONPATH:", os.environ.get("PYTHONPATH", "NA"))

        fs = fsspec.filesystem("file")
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        preprocessed_output = f"{tests_path}/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp", True)

        sdf = _raw_mdc_uri_folder_to_preprocessed_spark_df(
            window_start_time,
            window_end_time,
            f"{tests_path}/unit/raw_mdc_data/",
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
    def test_mdc_preprocessor(self, mdc_preprocessor_test_setup):
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

    @pytest.mark.parametrize(
        "url_str, converted",
        [
            ("https://my_account.blob.core.windows.net/my_container/path/to/file", True),
            ("wasbs://my_container@my_account.blob.core.windows.net/path/to/file", True),
            ("abfss://my_container@my_account.dfs.core.windows.net/path/to/file", True),
            (
                "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name"
                "/datastores/my_datastore/paths/path/to/file",
                True
            ),
            ("azureml://datastores/my_datastore/paths/path/to/file", True),
            ("azureml:my_asset:my_version", False),
            ("file://path/to/file", False),
            ("path/to/file", False),
        ]
    )
    def test_convert_to_azureml_long_form(self, url_str: str, converted: bool):
        """Test convert_to_azureml_long_form()."""
        converted_path = _convert_to_azureml_long_form(url_str, "my_datastore", "my_sub_id",
                                                       "my_rg_name", "my_ws_name")
        azureml_long = "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name" \
            "/datastores/my_datastore/paths/path/to/file"
        expected_path = azureml_long if converted else url_str
        assert converted_path == expected_path

    @pytest.mark.parametrize(
        "input_path, expected_datastore",
        [
            (
                "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name"
                "/datastores/long_form_datastore/paths/path/to/file",
                "long_form_datastore"
            ),
            ("azureml://datastores/short_form_datastore/paths/path/to/file", "short_form_datastore"),
            ("file://path/to/folder", None),
        ]
    )
    def test_get_datastore_from_input_path(self, input_path, expected_datastore):
        """Test get_datastore_from_input_path()."""
        datastore = _get_datastore_from_input_path(input_path)
        assert datastore == expected_datastore

    @pytest.mark.parametrize(
        "input_path",
        [
            "wasbs://my_container@my_account.blob.core.windows.net/path/to/file",
            "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
        ]
    )
    def test_get_datastore_from_input_path_throw_error(self, input_path):
        """Test get_datastore_from_input_path() with invalid input."""
        with pytest.raises(ValueError):
            _get_datastore_from_input_path(input_path)

    @pytest.mark.parametrize(
        "datastore, path, expected_datastore",
        [
            ("asset_datastore", None, "asset_datastore"),
            (
                None,
                "azureml://subscriptions/my_sub_id/resourcegroups/my_rg_name/workspaces/my_ws_name"
                "/datastores/long_form_datastore/paths/path/to/folder",
                "long_form_datastore"
            ),
            (None, "azureml://datastores/short_form_datastore/paths/path/to/folder", "short_form_datastore"),
            # (None, "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder", "workspaceblobstore")
        ]
    )
    def test_get_datastore_from_input_path_with_asset_path(self, datastore, path, expected_datastore):
        """Test get_datastore_from_input_path() with asset path."""
        mock_data_asset = Mock(datastore=datastore, path=path)
        mock_ml_client = Mock()
        mock_ml_client.data.get.return_value = mock_data_asset

        datastore = _get_datastore_from_input_path("azureml:my_asset:my_version", mock_ml_client)
        assert datastore == expected_datastore
