# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for mdc preprocessor."""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, DoubleType, LongType, ArrayType, MapType
import pytest
from unittest.mock import Mock
import fsspec
import shutil
import json
import os
import sys
import spark_mltable  # noqa, to enable spark.read.mltable
import pandas as pd
from pandas.testing import assert_frame_equal
from model_data_collector_preprocessor.run import (
    _raw_mdc_uri_folder_to_preprocessed_spark_df,
    _extract_data_and_correlation_id,
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
    python_path = sys.executable
    os.environ["PYSPARK_PYTHON"] = python_path
    print("PYSPARK_PYTHON", os.environ.get("PYSPARK_PYTHON", "NA"))
    module_path = f"{os.getcwd()}/src"
    old_python_path = os.environ.get("PYTHONPATH", None)
    old_python_path = f"{old_python_path};" if old_python_path else ""
    os.environ["PYTHONPATH"] = f"{old_python_path}{module_path}"
    print("PYTHONPATH:", os.environ.get("PYTHONPATH", "NA"))
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
            'sepal_length': [1, 2, 3, 1.5],
            'sepal_width': [2.3, 3.2, 3.4, 1.0],
            'petal_length': [2, 3, 3.2, 4],
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

    @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
    @pytest.mark.parametrize(
        "window_start_time, window_end_time, extract_correlation_id",
        [
            # chat history
            ("2023-10-30T16:00:00", "2023-10-30T17:00:00", False),
            ("2023-10-30T16:00:00", "2023-10-30T17:00:00", True),
            # ("2023-10-24T22:00:00", "2023-10-24T23:00:00", False),
            # ("2023-10-24T22:00:00", "2023-10-24T23:00:00", True),
        ]
    )
    def test_uri_folder_to_spark_df_with_chat_history(
            self, mdc_preprocessor_test_setup,
            window_start_time, window_end_time, extract_correlation_id):
        """Test uri_folder_to_spark_df() with chat_history column."""
        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())

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
        # todo: assert dataframe content

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

    @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
    @pytest.mark.parametrize(
        "data, expected_pdf, expected_fields",
        [
            # single input in each row
            (
                [
                    [json.dumps([{"f0": "v0",  "f1": 1,    "f2": 2}]), "cid0"],
                    [json.dumps([{"f0": "v1",  "f1": 1.2,  "f2": 3}]), "cid1"],
                    [json.dumps([{"f0": "v2",  "f1": 2.3,  "f2": 4}]), "cid2"],
                ],
                pd.DataFrame([
                    {"f0": "v0",    "f1": 1.0,  "f2": 2,    "correlationid": "cid0_0"},
                    {"f0": "v1",    "f1": 1.2,  "f2": 3,    "correlationid": "cid1_0"},
                    {"f0": "v2",    "f1": 2.3,  "f2": 4,    "correlationid": "cid2_0"},
                ]),
                [
                    StructField("f0", StringType()), StructField("f1", DoubleType()), StructField("f2", LongType()),
                    # StructField("correlationid", StringType(), False)
                ]
            ),
            # multiple inputs in one row
            (
                [
                    [json.dumps([{"f0": "v0",   "f1": 1,    "f2": 2},
                                 {"f0": "v3",   "f1": 1.5,  "f2": 5}]), "cid0"],
                    [json.dumps([{"f0": "v1",   "f1": 2,    "f2": 3}]), "cid1"],
                    [json.dumps([{"f0": "v2",   "f1": 3,    "f2": 4}]), "cid2"],
                ],
                pd.DataFrame([
                    {"f0": "v0",    "f1": 1.0,  "f2": 2,    "correlationid": "cid0_0"},
                    {"f0": "v3",    "f1": 1.5,  "f2": 5,    "correlationid": "cid0_1"},
                    {"f0": "v1",    "f1": 2,    "f2": 3,    "correlationid": "cid1_0"},
                    {"f0": "v2",    "f1": 3,    "f2": 4,    "correlationid": "cid2_0"},
                ]),
                [
                    StructField("f0", StringType()), StructField("f1", DoubleType()), StructField("f2", LongType()),
                    # StructField("correlationid", StringType(), False)
                ]
            ),
            # struct fields
            (
                [
                    [json.dumps([{"simple_field": "v0", "struct_field": {"f0": "t0", "f1": "u0", "f2": "w0"}}]), "cid0"],  # noqa
                    [json.dumps([{"simple_field": "v1", "struct_field": {"f0": "t1", "f1": "u1"}},
                                 {"simple_field": "v2", "struct_field": {"f0": "t2", "f1": "u2", "f2": "w2"}}]), "cid1"],  # noqa
                    [json.dumps([{"simple_field": "v3", "struct_field": {"f0": "t3",             "f2": "w3"}}]), "cid2"],  # noqa
                ],
                pd.DataFrame([
                    {"simple_field": "v0", "struct_field": {"f0": "t0", "f1": "u0", "f2": "w0"}, "correlationid": "cid0_0"},  # noqa
                    {"simple_field": "v1", "struct_field": {"f0": "t1", "f1": "u1"},             "correlationid": "cid1_0"},  # noqa
                    {"simple_field": "v2", "struct_field": {"f0": "t2", "f1": "u2", "f2": "w2"}, "correlationid": "cid1_1"},  # noqa
                    {"simple_field": "v3", "struct_field": {"f0": "t3",             "f2": "w3"}, "correlationid": "cid2_0"},  # noqa
                ]),
                [
                    StructField("simple_field", StringType()),
                    StructField("struct_field", MapType(StringType(), StringType())),
                    # StructField("correlationid", StringType(), False)
                ]
            ),
            # chat history
            (
                [
                    [
                        json.dumps([{"question": "q0", "chat_history": []}]),
                        "cid0"
                    ],
                    [
                        json.dumps([
                            {
                                "question": "q1",
                                "chat_history": [
                                    {
                                        "inputs": {"question": "q0"},
                                        "outputs": {"output": "o0"},
                                    }
                                ]
                            }
                        ]),
                        "cid1"
                    ],
                    [
                        json.dumps([
                            {
                                "question": "q2",
                                "chat_history": [
                                    {
                                        "inputs": {"question": "q0"},
                                        "outputs": {"output": "o0"},
                                    },
                                    {
                                        "inputs": {"question": "q1"},
                                        "outputs": {"output": "o1"},
                                    }
                                ]
                            }
                        ]),
                        "cid2"
                    ],
                ],
                pd.DataFrame([
                    {"question": "q0", "chat_history": [], "correlationid": "cid0_0"},
                    {
                        "question": "q1",
                        "chat_history": [
                            {
                                "inputs": {"question": "q0"},
                                "outputs": {"output": "o0"},
                            }
                        ],
                        "correlationid": "cid1_0"
                    },
                    {
                        "question": "q2",
                        "chat_history": [
                            {
                                "inputs": {"question": "q0"},
                                "outputs": {"output": "o0"},
                            },
                            {
                                "inputs": {"question": "q1"},
                                "outputs": {"output": "o1"},
                            }
                        ],
                        "correlationid": "cid2_0"
                    }
                ]),
                [
                    StructField("question", StringType()),
                    # StructField('chat_history', ArrayType(MapType(StringType(), MapType(StringType(), StringType())))),
                    # StructField("correlationid", StringType(), False)
                ]
            )
        ]
    )
    def test_extract_data_and_correlation_id(self, mdc_preprocessor_test_setup,
                                             data, expected_pdf, expected_fields):
        spark = SparkSession.builder.appName("test_extract_data_and_correlation_id").getOrCreate()
        expected_pdf.drop(columns=["chat_history"], inplace=True)
        extract_correlation_ids = [True, False]
        for extract_correlation_id in extract_correlation_ids:
            in_df = spark.createDataFrame(data, ["data", "correlationid"])
            out_df = _extract_data_and_correlation_id(in_df, extract_correlation_id)
            out_df.show(truncate=False)
            out_df.printSchema()
            fields = out_df.schema.fields
            for field in expected_fields:
                assert field in fields
            expected_pdf_ = expected_pdf
            if extract_correlation_id:
                assert StructField("correlationid", StringType(), False) in fields
            else:
                expected_pdf_ = expected_pdf.drop(columns=["correlationid"], inplace=False)
            actual_pdf = out_df.toPandas()
            assert_frame_equal(actual_pdf, expected_pdf_)

        # assert False

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
