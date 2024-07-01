# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for Gen AI preprocessor."""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField, StringType, TimestampType, StructType
)
import pytest
import os
import sys
from datetime import datetime
<<<<<<< HEAD
from model_data_collector_preprocessor.trace_aggregator import (
    aggregate_spans_into_traces,
)
=======
from model_data_collector_preprocessor.genai_preprocessor_df_schemas import _get_aggregated_trace_log_spark_df_schema
from model_data_collector_preprocessor.trace_aggregator import (
    aggregate_spans_into_traces,
)
from tests.unit.utils.unit_test_utils import assert_spark_dataframe_equal
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
from spark_mltable import SPARK_ZIP_PATH


@pytest.fixture(scope="module")
def genai_preprocessor_test_setup():
    # TODO: move this test utility to shared conftest later.
    # Check gsq tests and mdc tests for possible duplications.
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
    os.environ["PYTHONPATH"] = old_python_path  # change python path back to original


@pytest.mark.unit
class TestGenAISparkPreprocessor:
    """Test class for Gen AI Preprocessor."""

    def _init_spark(self) -> SparkSession:
        """Create spark session for tests."""
        spark: SparkSession = SparkSession.builder.appName("test").getOrCreate()
        sc = spark.sparkContext
        # if SPARK_ZIP_PATH is set, add the zip file to the spark context
        zip_path = os.environ.get(SPARK_ZIP_PATH, '')
        print(f"The spark_zip in environment: {zip_path}")
        if zip_path:
            sc.addPyFile(zip_path)
        return spark

    _preprocessed_log_schema = StructType([
        # TODO: The user_id and session_id may not be available in v1.
        StructField('attributes', StringType(), False),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
    ])

    _span_log_data = [
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"]
    ]

    # pass data that might have extra information like context, resource, etc
    _preprocessed_log_schema_extra = StructType([
        # TODO: The user_id and session_id may not be available in v1.
        StructField('attributes', StringType(), False),
        StructField('context', StringType(), True),
        StructField('c_resources', StringType(), True),
        StructField('end_time', TimestampType(), False),
        StructField('events', StringType(), False),
        StructField('framework', StringType(), False),
        StructField('links', StringType(), False),
        StructField('name', StringType(), False),
        StructField('parent_id', StringType(), True),
        # StructField('session_id', StringType(), True),
        StructField('span_id', StringType(), False),
        StructField('span_type', StringType(), False),
        StructField('start_time', TimestampType(), False),
        StructField('status', StringType(), False),
        StructField('trace_id', StringType(), False),
        # StructField('user_id', StringType(), True),
    ])

    _span_log_data_extra = [
        ['{"inputs":"in", "output":"out"}', "c", "{}"] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}', "c", "{}"] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}', "c", "{}"] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}', "c", "{}"] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"]
    ]

    _root_span_str_extra = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "context": "c", "c' + \
        '_resour''ces": "{}", "end_time": "2024-02-05T00:08:00", "events": "[]", "framework": "FLOW", "links": "' + \
        '[]", "name": "name", "parent_id": null, "span_id": "1", "span_type": "llm", "start_time": "2024-02-05T0' + \
        '0:01:00", "status": "OK", "trace_id": "01", "children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"out' + \
        'put\\":\\"out\\"}", "context": "c", "c_resources": "{}", "end_time": "2024-02-05T00:05:00", "events": "' + \
        '[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "1", "span_id": "2", "span_type": ' + \
        '"llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "children": [{"attributes' + \
        '": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "context": "c", "c_resources": "{}", "end_time": ' + \
        '"2024-02-05T00:04:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_' + \
        'id": "2", "span_id": "3", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "tra' + \
        'ce_id": "01", "children": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "con' + \
        'text": "c", "c_resources": "{}", "end_time": "2024-02-05T00:07:00", "events": "[]", "framework": "LLM",' + \
        ' "links": "[]", "name": "name", "parent_id": "1", "span_id": "4", "span_type": "llm", "start_time": "20' + \
        '24-02-05T00:06:00", "status": "OK", "trace_id": "01", "children": []}]}'

    _root_span_str = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": null, "span' + \
        '_id": "1", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "1", "span_id' + \
        '": "2", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "2", "span_' + \
        'id": "3", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "1", "sp' + \
        'an_id": "4", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _trace_log_data = [
            ["01", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str],
    ]

<<<<<<< HEAD
    _trace_log_schema = StructType(
        [
            StructField("trace_id", StringType(), False),
            StructField("user_id", StringType(), True),
            StructField("session_id", StringType(), True),
            StructField("start_time", TimestampType(), False),
            StructField("end_time", TimestampType(), False),
            StructField("input", StringType(), False),
            StructField("output", StringType(), False),
            StructField("root_span", StringType(), True),
        ]
    )
=======
    _trace_log_schema = _get_aggregated_trace_log_spark_df_schema()
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa

    _trace_log_data_extra = [
            ["01", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_extra],
    ]

    # 6-7 is desired hour
    # trace 00 is before the desired time.
    # trace 01 has child in hour before
    # trace 02
    _span_log_data_lookback = [
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 5, 10), "[]", "FLOW", "[]", "name"] +
        [None, "1", "llm", datetime(2024, 2, 5, 5, 15, 0), "OK", "00"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 5, 59), "[]", "RAG", "[]", "name"] +
        ["1", "2", "llm", datetime(2024, 2, 5, 5, 58, 0), "OK", "00"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 5, 59, 0), "[]", "INTERNAL", "[]", "name"] +
        ["4", "3", "llm", datetime(2024, 2, 5, 5, 58, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 5, 58, 0), "[]", "LLM", "[]", "name"] +
        ["5", "4", "llm", datetime(2024, 2, 5, 6, 5, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 6, 59, 0), "[]", "FLOW", "[]", "name"] +
        [None, "5", "llm", datetime(2024, 2, 5, 6, 8, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 6, 58, 0), "[]", "RAG", "[]", "name"] +
        ["6", "6", "llm", datetime(2024, 2, 5, 6, 50, 0), "OK", "02"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 7, 4, 0), "[]", "INTERNAL", "[]", "name"] +
        ["8", "7", "llm", datetime(2024, 2, 5, 6, 59, 0), "OK", "03"],
        ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 7, 7, 0), "[]", "LLM", "[]", "name"] +
        ["9", "8", "llm", datetime(2024, 2, 5, 7, 6, 0), "OK", "03"]
    ]

    _root_span_str_lookback = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "20' + \
        '24-02-05T06:59:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": nu' + \
        'll, "span_id": "5", "span_type": "llm", "start_time": "2024-02-05T06:08:00", "status": "OK", "trace_id"' + \
        ': "01", "children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "202' + \
        '4-02-05T05:58:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "5",' + \
        ' "span_id": "4", "span_type": "llm", "start_time": "2024-02-05T06:05:00", "status": "OK", "trace_id": "' + \
        '01", "children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-0' + \
        '2-05T05:59:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "4' + \
        '", "span_id": "3", "span_type": "llm", "start_time": "2024-02-05T05:58:00", "status": "OK", "trace_id":' + \
        ' "01", "children": []}]}]}'

    _trace_log_data_lookback = [
            ["01", None, None, datetime(2024, 2, 5, 6, 8, 0)] +
            [datetime(2024, 2, 5, 6, 59, 0), "in", "out", _root_span_str_lookback],
    ]

    _span_log_data_request_id = [
        ['{"inputs":"in", "output":"out", "request_id":"0x01"}'] +
        [datetime(2024, 2, 5, 9, 30), "[]", "FLOW", "[]", "name"] +
        [None, "1", "llm", datetime(2024, 2, 5, 9, 15), "OK", "00"],
        ['{"inputs":"in", "output":"out", "request_id":"0x02"}'] +
        [datetime(2024, 2, 5, 9, 55), "[]", "RAG", "[]", "name"] +
        [None, "2", "llm", datetime(2024, 2, 5, 9, 6), "OK", "00"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 9, 59), "[]", "INTERNAL", "[]", "name"] +
        ["4", "3", "llm", datetime(2024, 2, 5, 9, 58), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 9, 58), "[]", "LLM", "[]", "name"] +
        ["5", "4", "llm", datetime(2024, 2, 5, 9, 5), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 9, 59), "[]", "FLOW", "[]", "name"] +
        [None, "5", "llm", datetime(2024, 2, 5, 9, 8), "OK", "01"],
    ]

    _root_span_str_request_0x01 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\", \\"request_id' + \
        '\\":\\"0x01\\"}", "end_time": "2024-02-05T09:30:00", "events": "[]", "framework": "FLOW", "links": "[]"' + \
        ', "name": "name", "parent_id": null, "span_id": "1", "span_type": "llm", "start_time": "2024-02-05T09:1' + \
        '5:00", "status": "OK", "trace_id": "0x01", "children": []}'

    _root_span_str_request_0x02 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\", \\"request_id' + \
        '\\":\\"0x02\\"}", "end_time": "2024-02-05T09:55:00", "events": "[]", "framework": "RAG", "links": "[]",' + \
        ' "name": "name", "parent_id": null, "span_id": "2", "span_type": "llm", "start_time": "2024-02-05T09:06' + \
        ':00", "status": "OK", "trace_id": "0x02", "children": []}'

    _root_span_str_request_01 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "' + \
        '2024-02-05T09:59:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": ' + \
        'null, "span_id": "5", "span_type": "llm", "start_time": "2024-02-05T09:08:00", "status": "OK", "trace_i' + \
        'd": "01", "children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2' + \
        '024-02-05T09:58:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "5' + \
        '", "span_id": "4", "span_type": "llm", "start_time": "2024-02-05T09:05:00", "status": "OK", "trace_id":' + \
        ' "01", "children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024' + \
        '-02-05T09:59:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": ' + \
        '"4", "span_id": "3", "span_type": "llm", "start_time": "2024-02-05T09:58:00", "status": "OK", "trace_id' + \
        '": "01", "children": []}]}]}'

    _trace_log_data_request_id = [
            ["0x01", None, None, datetime(2024, 2, 5, 9, 15)] +
            [datetime(2024, 2, 5, 9, 30), "in", "out", _root_span_str_request_0x01],
            ["0x02", None, None, datetime(2024, 2, 5, 9, 6)] +
            [datetime(2024, 2, 5, 9, 55), "in", "out", _root_span_str_request_0x02],
            ["01", None, None, datetime(2024, 2, 5, 9, 8)] +
            [datetime(2024, 2, 5, 9, 59), "in", "out", _root_span_str_request_01],
    ]

<<<<<<< HEAD
=======
    _span_log_data_same_trace = [
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", "00"] +
        ["5", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "5"] +
        ["6", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "6"] +
        ["7", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "5"] +
        ["8", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["9", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "9"] +
        ["10", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "10"] +
        ["11", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "9"] +
        ["12", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"],
    ]

    _root_span_str_0 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": null, "span' + \
        '_id": "1", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01_0", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "1", "span_id' + \
        '": "2", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "2", "span_' + \
        'id": "3", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "1", "sp' + \
        'an_id": "4", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _root_span_str_1 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": "00", "span' + \
        '_id": "5", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01_1", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "5", "span_id' + \
        '": "6", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "6", "span_' + \
        'id": "7", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "5", "sp' + \
        'an_id": "8", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _root_span_str_2 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": null, "span' + \
        '_id": "9", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01_2", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "9", "span_id' + \
        '": "10", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "10", "span_' + \
        'id": "11", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "9", "sp' + \
        'an_id": "12", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _trace_log_data_same_trace = [
            ["01_0", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_0],
            ["01_1", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_1],
            ["01_2", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_2],
    ]

    _span_log_data_with_error = [
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "1"] +
        ["2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "2"] +
        ["3", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "1"] +
        ["4", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"],
        ['{"inputs":"in"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", "00"] +
        ["5", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "5"] +
        ["6", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "6"] +
        ["7", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "5"] +
        ["8", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["9", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "9"] +
        ["10", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "10"] +
        ["11", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "9"] +
        ["12", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name", None] +
        ["13", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name", "13"] +
        ["14", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 4, 0), "[]", "INTERNAL", "[]", "name", "14"] +
        ["15", "llm", datetime(2024, 2, 5, 0, 3, 0), "OK", "01"],
        ['{"inputs":"in", "output":"out"}'] +
        [datetime(2024, 2, 5, 0, 7, 0), "[]", "LLM", "[]", "name", "13"] +
        ["16", "llm", datetime(2024, 2, 5, 0, 6, 0), "OK", "01"]
    ]

    _root_span_str_error = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}",' + \
        ' "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": null, "span' + \
        '_id": "1", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01_0", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "1", "span_id' + \
        '": "2", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "2", "span_' + \
        'id": "3", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "1", "sp' + \
        'an_id": "4", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _root_span_str_error3 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}",' + \
        ' "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": null, "span' + \
        '_id": "9", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01_2", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "9", "span_id' + \
        '": "10", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "10", "span_' + \
        'id": "11", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "9", "sp' + \
        'an_id": "12", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _root_span_str_error4 = '{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}",' + \
        ' "end_time": "2024-02-05T' + \
        '00:08:00", "events": "[]", "framework": "FLOW", "links": "[]", "name": "name", "parent_id": null, "span_' + \
        'id": "13", "span_type": "llm", "start_time": "2024-02-05T00:01:00", "status": "OK", "trace_id": "01_3", "' + \
        'children": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T0' + \
        '0:05:00", "events": "[]", "framework": "RAG", "links": "[]", "name": "name", "parent_id": "13", "span_id' + \
        '": "14", "span_type": "llm", "start_time": "2024-02-05T00:02:00", "status": "OK", "trace_id": "01", "chi' + \
        'ldren": [{"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02-05T00:0' + \
        '4:00", "events": "[]", "framework": "INTERNAL", "links": "[]", "name": "name", "parent_id": "14", "span_' + \
        'id": "15", "span_type": "llm", "start_time": "2024-02-05T00:03:00", "status": "OK", "trace_id": "01", "c' + \
        'hildren": []}]}, {"attributes": "{\\"inputs\\":\\"in\\", \\"output\\":\\"out\\"}", "end_time": "2024-02' + \
        '-05T00:07:00", "events": "[]", "framework": "LLM", "links": "[]", "name": "name", "parent_id": "13", "sp' + \
        'an_id": "16", "span_type": "llm", "start_time": "2024-02-05T00:06:00", "status": "OK", "trace_id": "01",' + \
        ' "children": []}]}'

    _trace_log_data_with_error = [
            ["01_0", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_error],
            ["01_2", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_error3],
            ["01_3", None, None, datetime(2024, 2, 5, 0, 1, 0)] +
            [datetime(2024, 2, 5, 0, 8, 0), "in", "out", _root_span_str_error4]
    ]

>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
    def test_trace_aggregator_empty_root_span(self, code_zip_test_setup, genai_preprocessor_test_setup):
        """Test scenarios where we have a faulty root span when generating tree."""
        spark = self._init_spark()
        start_time = datetime(2024, 2, 5, 0,)
        end_time = datetime(2024, 2, 5, 1)

        span_logs_no_root_with_data = [
            ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name"] +
            [None, "1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
            ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name"] +
            ["1", "2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "02"],
        ]
        span_logs_no_root_with_data_df = spark.createDataFrame(
            span_logs_no_root_with_data,
            self._preprocessed_log_schema)

        trace_df = aggregate_spans_into_traces(span_logs_no_root_with_data_df, True, start_time, end_time)
        rows = trace_df.collect()
<<<<<<< HEAD
        assert trace_df.count() == 1
        assert rows[0]['trace_id'] == "01"

        span_logs_no_root = [
            ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 0, 8, 0), "[]", "FLOW", "[]", "name",] +
            ["1", "1", "llm", datetime(2024, 2, 5, 0, 1, 0), "OK", "01"],
        ]
        spans_no_root_df = spark.createDataFrame(span_logs_no_root, self._preprocessed_log_schema)
=======
        assert trace_df.count() == 2
        assert rows[0]['trace_id'] == "01"
        assert rows[1]['trace_id'] == "02"

        span_logs_no_root_forest = [
            ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name"] +
            ["1", "1", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
            ['{"inputs":"in", "output":"out"}', datetime(2024, 2, 5, 0, 5, 0), "[]", "RAG", "[]", "name"] +
            ["1", "2", "llm", datetime(2024, 2, 5, 0, 2, 0), "OK", "01"],
        ]
        spans_no_root_df = spark.createDataFrame(span_logs_no_root_forest, self._preprocessed_log_schema)
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
        no_root_traces = aggregate_spans_into_traces(spans_no_root_df, True, start_time, end_time)
        assert no_root_traces.isEmpty()

    @pytest.mark.parametrize(
        "span_input_logs, span_input_schema, expected_trace_logs, " +
        "expected_trace_schema, require_trace_data, data_window_start, data_window_end",
        [
            ([], _preprocessed_log_schema, [], _trace_log_schema, True,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            (_span_log_data, _preprocessed_log_schema, _trace_log_data, _trace_log_schema, True,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            (_span_log_data_extra, _preprocessed_log_schema_extra, _trace_log_data_extra, _trace_log_schema, True,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            (_span_log_data, _preprocessed_log_schema, [], _trace_log_schema, False,
             datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            # Look back with extra span logs
            (_span_log_data_lookback, _preprocessed_log_schema, _trace_log_data_lookback, _trace_log_schema, True,
             datetime(2024, 2, 5, 6), datetime(2024, 2, 5, 7)),
            # request_id data
<<<<<<< HEAD
            (_span_log_data_request_id, _preprocessed_log_schema, _trace_log_data_request_id, _trace_log_schema, True,
             datetime(2024, 2, 5, 9), datetime(2024, 2, 5, 10))
=======
            # TODO: uncomment when we want to test replacing trace_id with request_id
            # (_span_log_data_request_id, _preprocessed_log_schema, _trace_log_data_request_id, _trace_log_schema,
            #  True, datetime(2024, 2, 5, 9), datetime(2024, 2, 5, 10))
            (_span_log_data_same_trace, _preprocessed_log_schema, _trace_log_data_same_trace, _trace_log_schema,
             True, datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
            # one of the logs won't have input or output
            (_span_log_data_with_error, _preprocessed_log_schema, _trace_log_data_with_error, _trace_log_schema,
             True, datetime(2024, 2, 5, 0), datetime(2024, 2, 5, 1)),
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
        ]
    )
    def test_trace_aggregator(
            self, code_zip_test_setup, genai_preprocessor_test_setup,
            span_input_logs, span_input_schema, expected_trace_logs, expected_trace_schema,
            require_trace_data, data_window_start, data_window_end):
        """Test scenario where spans has real data."""
        spark = self._init_spark()

        # infer schema only when we have data.
        processed_spans_df = spark.createDataFrame(span_input_logs, span_input_schema)
        expected_traces_df = spark.createDataFrame(expected_trace_logs, expected_trace_schema)

        print("processed logs:")
        processed_spans_df.show()
        processed_spans_df.printSchema()

        print("expected trace logs:")
        expected_traces_df.show()
        expected_traces_df.printSchema()

        actual_trace_df = aggregate_spans_into_traces(
            processed_spans_df, require_trace_data, data_window_start, data_window_end)

        print("actual trace logs:")
        actual_trace_df.show()
        actual_trace_df.printSchema()

        assert_spark_dataframe_equal(actual_trace_df, expected_traces_df)
<<<<<<< HEAD


def assert_spark_dataframe_equal(df1, df2):
    """Assert two spark dataframes are equal."""
    assert df1.schema == df2.schema
    assert df1.count() == df2.count()
    print(f'df1: {df1.collect()}')
    print(f'df2: {df2.collect()}')
    assert df1.collect() == df2.collect()
=======
>>>>>>> 7a54b91f3a492ed00e3033a99450bbc4df36a0fa
