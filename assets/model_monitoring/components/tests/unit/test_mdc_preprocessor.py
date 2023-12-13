# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for mdc preprocessor."""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, DoubleType, LongType, BooleanType
import pytest
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
)
from shared_utilities.momo_exceptions import DataNotFoundError


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

    @pytest.mark.skip(reason="pending on a mltable column to_string() bug")
    @pytest.mark.parametrize(
        "window_start_time, window_end_time",
        [
            ("2023-11-12T10:00:00", "2023-11-12T11:00:00"),
        ]
    )
    def test_uri_folder_to_spark_df_with_complex_type(self, mdc_preprocessor_test_setup,
                                                      window_start_time, window_end_time):
        """Test uri_folder_to_spark_df()."""
        fs = fsspec.filesystem("file")
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        preprocessed_output = f"{tests_path}/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp", True)

        pdf = _raw_mdc_uri_folder_to_preprocessed_spark_df(
            window_start_time,
            window_end_time,
            f"{tests_path}/unit/raw_mdc_data/",
            preprocessed_output,
            False,
            fs
        )
        pdf.show()

    @pytest.mark.parametrize(
        "window_start_time, window_end_time, root_folder_exists",
        [
            ("2023-11-03T15:00:00", "2023-11-03T16:00:00", True),  # no window folder
            ("2023-11-06T15:00:00", "2023-11-06T16:00:00", True),  # has window folder, no file
            ("2023-11-06T17:00:00", "2023-11-06T18:00:00", True),  # has window folder and file, but empty file
            ("2023-11-08T14:00:00", "2023-11-08T16:00:00", False),  # root folder not exists
        ]
    )
    def test_uri_folder_to_spark_df_no_data(self, mdc_preprocessor_test_setup,
                                            window_start_time, window_end_time, root_folder_exists):
        """Test uri_folder_to_spark_df()."""
        def my_add_tags(tags: dict):
            print("my_add_tags:", tags)
        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())

        fs = fsspec.filesystem("file")
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        preprocessed_output = f"{tests_path}/unit/preprocessed_mdc_data"
        shutil.rmtree(f"{preprocessed_output}temp", True)
        root_folder = f"{tests_path}/unit/raw_mdc_data/" if root_folder_exists else f"{tests_path}/unit/raw_mdc_data1/"

        with pytest.raises(DataNotFoundError):
            df = _raw_mdc_uri_folder_to_preprocessed_spark_df(
                window_start_time,
                window_end_time,
                root_folder,
                preprocessed_output,
                False,
                fs,
                my_add_tags
            )
            df.show()

    @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
    @pytest.mark.parametrize(
        "window_start_time, window_end_time, extract_correlation_id",
        [
            # chat history
            ("2023-10-30T16:00:00", "2023-10-30T17:00:00", False),
            ("2023-10-30T16:00:00", "2023-10-30T17:00:00", True),
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
                    [json.dumps([{"f0": "v0",  "f1": 1,    "f2": 2, "f3": True,  "f4": "2023-11-08T07:01:02Z"}]), "cid0"],  # noqa
                    [json.dumps([{"f0": "v1",  "f1": 1.2,  "f2": 3, "f3": False, "f4": "2023-11-08T07:02:03Z"}]), "cid1"],  # noqa
                    [json.dumps([{"f0": "v2",  "f1": 2.3,  "f2": 4, "f3": True,  "f4": "2023-11-08T08:00:05Z"}]), "cid2"],  # noqa
                ],
                pd.DataFrame([
                    {"f0": "v0",    "f1": 1.0,  "f2": 2,    "f3": True,     "f4": "2023-11-08T07:01:02Z",   "correlationid": "cid0_0"},  # noqa
                    {"f0": "v1",    "f1": 1.2,  "f2": 3,    "f3": False,    "f4": "2023-11-08T07:02:03Z",   "correlationid": "cid1_0"},  # noqa
                    {"f0": "v2",    "f1": 2.3,  "f2": 4,    "f3": True,     "f4": "2023-11-08T08:00:05Z",   "correlationid": "cid2_0"},  # noqa
                ]),
                [
                    StructField("f0", StringType()), StructField("f1", DoubleType()), StructField("f2", LongType()),
                    StructField("f3", BooleanType()), StructField("f4", StringType()),
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
                ]
            ),
            # struct fields, with escape characters
            (
                [
                    [json.dumps([{"simple_field": "v0", "struct_field": {"f0": "t\\0",  "f1": 1,    "f2": 4}}]), "cid0"],  # noqa
                    [json.dumps([{"simple_field": "v1", "struct_field": {"f0": "t\"1",  "f1": 1.2}},
                                 {"simple_field": "v2", "struct_field": {"f0": "\"t2\"","f1": 1.3,  "f2": 5}}]), "cid1"],  # noqa
                    [json.dumps([{"simple_field": "v3", "struct_field": {"f0": "\"[\\\"t3\\\"]\"",                "f2": 6}}]), "cid2"],  # noqa
                ],
                pd.DataFrame([
                    {"simple_field": "v0", "struct_field": json.dumps({"f0": "t\\0",    "f1": 1,   "f2": 4}),   "correlationid": "cid0_0"},  # noqa
                    {"simple_field": "v1", "struct_field": json.dumps({"f0": "t\"1",    "f1": 1.2}),            "correlationid": "cid1_0"},  # noqa
                    {"simple_field": "v2", "struct_field": json.dumps({"f0": "\"t2\"",  "f1": 1.3, "f2": 5}),   "correlationid": "cid1_1"},  # noqa
                    {"simple_field": "v3", "struct_field": json.dumps({"f0": "\"[\\\"t3\\\"]\"",                 "f2": 6}),   "correlationid": "cid2_0"},  # noqa
                ]),
                [
                    StructField("simple_field", StringType()),
                    StructField("struct_field", StringType()),
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
                    {"question": "q0", "chat_history": json.dumps([]), "correlationid": "cid0_0"},
                    {
                        "question": "q1",
                        "chat_history": json.dumps([
                            {
                                "inputs": {"question": "q0"},
                                "outputs": {"output": "o0"},
                            }
                        ]),
                        "correlationid": "cid1_0"
                    },
                    {
                        "question": "q2",
                        "chat_history": json.dumps([
                            {
                                "inputs": {"question": "q0"},
                                "outputs": {"output": "o0"},
                            },
                            {
                                "inputs": {"question": "q1"},
                                "outputs": {"output": "o1"},
                            }
                        ]),
                        "correlationid": "cid2_0"
                    }
                ]),
                [
                    StructField("question", StringType()),
                    StructField("chat_history", StringType()),
                    # StructField('chat_history',
                    #             ArrayType(MapType(StringType(), MapType(StringType(), StringType())))),
                ]
            )
        ]
    )
    def test_extract_data_and_correlation_id(self, mdc_preprocessor_test_setup,
                                             data, expected_pdf, expected_fields):
        """Test _extract_data_and_correlation_id()."""
        spark = SparkSession.builder.appName("test_extract_data_and_correlation_id").getOrCreate()
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
