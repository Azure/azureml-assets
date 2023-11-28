# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for mdc preprocessor."""

from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StringType, DoubleType, BooleanType, LongType, ArrayType, StructType
import pytest
import os
import sys
import json
from datetime import datetime
from model_data_collector_preprocessor.spark_run import (
    _mdc_uri_folder_to_raw_spark_df, _extract_data_and_correlation_id
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
class TestMDCSparkPreprocessor:
    """Test class for MDC Preprocessor."""

    _data_schema = ArrayType(StructType([
        StructField("petal_length", DoubleType()),
        StructField("petal_width", DoubleType()),
        StructField("sepal_length", DoubleType()),
        StructField("sepal_width", DoubleType()),
    ]))

    # @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
    @pytest.mark.parametrize(
        "window_start_time, window_end_time, expected_schema, expected_data",
        [
            # data only
            (
                datetime(2023, 10, 11, 20), datetime(2023, 10, 11, 21),
                StructType([
                    StructField("correlationid", StringType()),
                    StructField("data", _data_schema)
                ]),
                [
                    ["7f16d5b1-76f9-4b3e-b82d-fc21d29356a5", [(2.0, 1.3, 1.0, 2.3)]],
                    ["f2b524a7-3272-45df-a530-c945004de305", [(3.0, 1.5, 2.0, 3.2), (3.2, 1.8, 3.0, 3.4)]],
                    ["95e1afa0-256d-414b-8e4c-fea1baa98225", [(4.0, 1.6, 1.5, 1.0)]]
                ]
            ),
            # data and dataref mix
            (
                datetime(2023, 10, 15, 17), datetime(2023, 10, 15, 18),
                StructType([
                    StructField("correlationid", StringType()),
                    StructField("data", _data_schema),
                    StructField("dataref", StringType())
                ]),
                [
                    ["7f16d5b1-76f9-4b3e-b82d-fc21d29356a5", [(2.0, 1.3, 1.0, 2.3)], None],
                    ["f2b524a7-3272-45df-a530-c945004de305", None, "tests/unit/raw_mdc_data/2023/10/15/17/mdc_dataref_1.json"],  # noqa
                    ["95e1afa0-256d-414b-8e4c-fea1baa98225", [(4.0, 1.6, 1.5, 1.0)], None],
                ]
            ),
            # dataref only
            (
                datetime(2023, 10, 16, 21), datetime(2023, 10, 16, 22),
                StructType([
                    StructField("correlationid", StringType()),
                    StructField("dataref", StringType())
                ]),
                [
                    ["7f16d5b1-76f9-4b3e-b82d-fc21d29356a5", "tests/unit/raw_mdc_data/2023/10/16/21/mdc_dataref_0.json"],  # noqa
                    ["f2b524a7-3272-45df-a530-c945004de305", "tests/unit/raw_mdc_data/2023/10/16/21/mdc_dataref_1.json"],  # noqa
                    ["95e1afa0-256d-414b-8e4c-fea1baa98225", "tests/unit/raw_mdc_data/2023/10/16/21/mdc_dataref_2.json"],  # noqa
                ]
            ),
        ]
    )
    def test_uri_folder_to_raw_spark_df(self, mdc_preprocessor_test_setup, window_start_time, window_end_time,
                                        expected_schema, expected_data):
        """Test uri_folder_to_spark_df()."""
        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())

        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")

        df = _mdc_uri_folder_to_raw_spark_df(
            window_start_time,
            window_end_time,
            f"{tests_path}/unit/raw_mdc_data/"
        )
        columns = ["correlationid"]
        if "data" in df.columns:
            columns.append("data")
        if "dataref" in df.columns:
            columns.append("dataref")
        actual_df = df.select(columns)
        print("raw dataframe:")
        actual_df.show(truncate=False)
        actual_df.printSchema()

        expected_df = SparkSession.builder.getOrCreate().createDataFrame(expected_data, schema=expected_schema)
        expected_df.show(truncate=False)
        expected_df.printSchema()

        assert_spark_dataframe_equal(actual_df, expected_df)

    @pytest.mark.parametrize(
        "window_start_time, window_end_time, root_folder_exists",
        [
            (datetime(2023, 11, 3, 15), datetime(2023, 11, 3, 16), True),  # no window folder
            (datetime(2023, 11, 6, 15), datetime(2023, 11, 6, 16), True),  # has window folder, no file
            (datetime(2023, 11, 6, 17), datetime(2023, 11, 6, 18), True),  # has window folder and file, but empty file
            (datetime(2023, 11, 8, 14), datetime(2023, 11, 8, 16), False),  # root folder not exists
        ]
    )
    def test_uri_folder_to_raw_spark_df_no_data(self, mdc_preprocessor_test_setup,
                                                window_start_time, window_end_time, root_folder_exists):
        """Test uri_folder_to_spark_df()."""
        def my_add_tags(tags: dict):
            print("my_add_tags:", tags)

        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())

        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        root_folder = f"{tests_path}/unit/raw_mdc_data/" if root_folder_exists else f"{tests_path}/unit/raw_mdc_data1/"

        with pytest.raises(DataNotFoundError):
            df = _mdc_uri_folder_to_raw_spark_df(
                window_start_time,
                window_end_time,
                root_folder,
                my_add_tags
            )
            df.show()

    # @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
    @pytest.mark.parametrize(
        "window_start_time, window_end_time",
        [
            # chat history
            (datetime(2023, 10, 30, 16), datetime(2023, 10, 30, 17)),
        ]
    )
    def test_uri_folder_to_raw_spark_df_with_chat_history(self, mdc_preprocessor_test_setup,
                                                          window_start_time, window_end_time):
        """Test uri_folder_to_spark_df() with chat_history column."""
        print("testing test_uri_folder_to_spark_df...")
        print("working dir:", os.getcwd())

        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")

        df = _mdc_uri_folder_to_raw_spark_df(
            window_start_time,
            window_end_time,
            f"{tests_path}/unit/raw_mdc_data/",
        )
        df = df.select("correlationid", "data")
        print("preprocessed dataframe:")
        df.show(truncate=False)
        df.printSchema()
        # todo: assert dataframe content

    # @pytest.mark.skip(reason="can't set PYTHONPATH for executor in remote run.")
    @pytest.mark.parametrize(
        "raw_data, expected_data, expected_schema",
        [
            # single input in each row
            (
                [
                    json.dumps({
                        "data": [{"f0": "v0", "f1": 1, "f2": 2, "f3": True, "f4": "2023-11-08T07:01:02Z"}],
                        "correlationid": "cid0"
                    }),
                    json.dumps({
                        "data": [{"f0": "v1", "f1": 1.2, "f2": 3, "f3": False, "f4": "2023-11-08T07:02:03Z"}],
                        "correlationid": "cid1"
                    }),
                    json.dumps({
                        "data": [{"f0": "v2", "f1": 2.3, "f2": 4, "f3": True, "f4": "2023-11-08T08:00:05Z"}],
                        "correlationid": "cid2"
                    }),
                ],
                [
                    ["v0", 1.0, 2, True,  "2023-11-08T07:01:02Z", "cid0_0"],
                    ["v1", 1.2, 3, False, "2023-11-08T07:02:03Z", "cid1_0"],
                    ["v2", 2.3, 4, True,  "2023-11-08T08:00:05Z", "cid2_0"],
                ],
                StructType([
                    StructField("f0", StringType()),
                    StructField("f1", DoubleType()),
                    StructField("f2", LongType()),
                    StructField("f3", BooleanType()),
                    StructField("f4", StringType()),
                    StructField("correlationid", StringType(), False),
                ])
            ),
            # multiple inputs in one row
            (
                [
                    json.dumps({"data": [{"f0": "v00", "f1": 1,   "f2": 2},
                                         {"f0": "v01", "f1": 1.5, "f2": 5}], "correlationid": "cid0"}),
                    json.dumps({"data": [{"f0": "v1",  "f1": 2,   "f2": 3}], "correlationid": "cid1"}),
                    json.dumps({"data": [{"f0": "v2",  "f1": 3,   "f2": 4}], "correlationid": "cid2"}),
                ],
                [
                    ["v00", 1.0, 2, "cid0_0"],
                    ["v01", 1.5, 5, "cid0_1"],
                    ["v1",  2.0, 3, "cid1_0"],
                    ["v2",  3.0, 4, "cid2_0"],
                ],
                StructType([
                    StructField("f0", StringType()),
                    StructField("f1", DoubleType()),
                    StructField("f2", LongType()),
                    StructField("correlationid", StringType(), False),
                ])
            ),
            # struct fields, with escape characters
            (
                [
                    json.dumps({
                        "data": [{"simple_field": "v0", "struct_field": {"f0": r"t\0", "f1": [], "f2": 4}}],
                        "correlationid": "cid0"
                    }),
                    json.dumps({
                        "data": [{"simple_field": "v1", "struct_field": {"f0": 't"1',  "f1": [1]}},
                                 {"simple_field": "v2", "struct_field": {"f0": '"t2"', "f1": [1, 2], "f2": 5}}],
                        "correlationid": "cid1"
                    }),
                    json.dumps({
                        "data": [{
                            "simple_field": "v3", "struct_field": {"f0": r'"[\"t3\"]"', "f1": [1, 2, 3], "f2": 6}
                        }],
                        "correlationid": "cid2"
                    }),
                ],
                [
                    ["v0", (r"t\0",        [],        4),    "cid0_0"],
                    ["v1", ('t"1',         [1],       None), "cid1_0"],
                    ["v2", ('"t2"',        [1, 2],    5),    "cid1_1"],
                    ["v3", (r'"[\"t3\"]"', [1, 2, 3], 6),    "cid2_0"],
                ],
                StructType([
                    StructField("simple_field", StringType()),
                    StructField("struct_field", StructType([
                        StructField("f0", StringType()),
                        StructField("f1", ArrayType(LongType())),
                        StructField("f2", LongType())
                    ])),
                    StructField("correlationid", StringType(), False),
                ])
            ),
            # chat history
            (
                [
                    json.dumps({"data": [{"question": "q0", "chat_history": []}], "correlationid": "cid0"}),
                    json.dumps({
                        "data": [
                            {
                                "question": "q1",
                                "chat_history": [
                                    {
                                        "inputs": {"question": "q0"},
                                        "outputs": {"output": "o0"},
                                    }
                                ]
                            }
                        ],
                        "correlationid": "cid1"
                    }),
                    json.dumps({
                        "data": [
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
                        ],
                        "correlationid": "cid2"
                    }),
                ],
                [
                    [[], "q0", "cid0_0"],
                    [
                        [
                            {
                                "inputs": {"question": "q0"},
                                "outputs": {"output": "o0"},
                            }
                        ],
                        "q1", "cid1_0"
                    ],
                    [
                        [
                            {
                                "inputs": {"question": "q0"},
                                "outputs": {"output": "o0"},
                            },
                            {
                                "inputs": {"question": "q1"},
                                "outputs": {"output": "o1"},
                            }
                        ],
                        "q2", "cid2_0"
                    ]
                ],
                StructType([
                    StructField("chat_history", ArrayType(
                        StructType([
                            StructField("inputs", StructType([
                                StructField("question", StringType())
                            ])),
                            StructField("outputs", StructType([
                                StructField("output", StringType())
                            ]))
                        ])
                    )),
                    StructField("question", StringType()),
                    StructField("correlationid", StringType(), False),
                ])
            )
        ]
    )
    def test_extract_data_and_correlation_id(self, mdc_preprocessor_test_setup,
                                             raw_data, expected_data, expected_schema):
        """Test _extract_data_and_correlation_id()."""
        spark = SparkSession.builder.appName("test_extract_data_and_correlation_id").getOrCreate()
        sc = spark.sparkContext
        extract_correlation_ids = [True, False]
        for extract_correlation_id in extract_correlation_ids:
            rdd = sc.parallelize(raw_data)
            in_df = spark.read.json(rdd)
            out_df = _extract_data_and_correlation_id(in_df, extract_correlation_id)
            out_df.show(truncate=False)
            out_df.printSchema()

            expected_df = spark.createDataFrame(expected_data, schema=expected_schema)
            if not extract_correlation_id:
                expected_df = expected_df.drop("correlationid")

            assert_spark_dataframe_equal(out_df, expected_df)


def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    assert df1.collect() == df2.collect()
