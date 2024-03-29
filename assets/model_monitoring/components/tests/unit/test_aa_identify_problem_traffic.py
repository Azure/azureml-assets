# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""test class for action analyzer identify problem traffic."""

from pyspark.sql import SparkSession
from pyspark.sql.types import (
    StructField, StringType, TimestampType, StructType
)
import pytest
import os
import sys
import json
from datetime import datetime
from src.action_analyzer.action_analyzer_identify_problem_traffic.run import (
    add_query_intention,
    get_violated_metrics,
    get_query_intention
)
from src.action_analyzer.constants import (
    PROMPT_COLUMN,
    QUERY_INTENTION_COLUMN,
    DEFAULT_TOPIC_NAME
)
from pyspark.sql.functions import col, lit
from src.model_data_collector_preprocessor.store_url import StoreUrl
import spark_mltable  # noqa, to enable spark.read.mltable
from spark_mltable import SPARK_ZIP_PATH


@pytest.mark.unit
class TestActionAnalyzerIdentifyProblemTraffic:
    """Test class for action analyzer identify problem traffic."""

    def _init_spark(self) -> SparkSession:
        """Create spark session for tests."""
        return SparkSession.builder.appName("test").getOrCreate()

    def test_get_violated_metrics_success(self):
        """Test get violated metrics from gsq output."""
        # "AggregatedFluencyPassRate": {
        #     "value": "0.9411764705882353",
        #     "threshold": "0.9",
        # }
        # "AggregatedCoherencePassRate": {
        #     "value": "0.8627450980392157",
        #     "threshold": "0.9",
        # }
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        input_url = f"{tests_path}/unit/resources/"
        violated_metrics = get_violated_metrics(input_url, "gsq-signal")
        assert violated_metrics == ["Coherence"]

    def test_get_violated_metrics_fail(self):
        """Test get violated metrics from empty gsq output."""
        tests_path = os.path.abspath(f"{os.path.dirname(__file__)}/../../tests")
        input_url = f"{tests_path}/unit/resources/"
        violated_metrics = get_violated_metrics(input_url, "empty")
        assert violated_metrics == []

    def test_add_query_intention_small_group_size(self):
        """Test add query intention when the data size is smaller than the minimum group size."""
        data = [["q1"], ["q2"], ["q3"], ["q4"]]
        data_schema = StructType(
            [
                StructField(PROMPT_COLUMN, StringType(), True)
            ]
        )
        spark = self._init_spark()
        data_df = spark.createDataFrame(data, schema=data_schema)
        query_intention_df = add_query_intention(data_df, "workspace_connection_id", "model_deployment_name", True)
        query_intentions = query_intention_df.select(QUERY_INTENTION_COLUMN).rdd.flatMap(lambda x: x).collect()
        assert query_intentions == [DEFAULT_TOPIC_NAME] * 4

    def test_add_query_intention_error_in_bertopic(self):
        """Test add query intention when Bertopic throw exception."""
        data = [
                    ["q1"], ["q2"], ["q3"], ["q4"],
                    ["q5"], ["q6"], ["q7"], ["q8"],
                    ["q9"], ["q10"], ["q11"], ["q12"]
                ]
        data_schema = StructType(
            [
                StructField(PROMPT_COLUMN, StringType(), True)
            ]
        )
        spark = self._init_spark()
        data_df = spark.createDataFrame(data, schema=data_schema)
        query_intention_df = add_query_intention(data_df, "workspace_connection_id", "model_deployment_name", True)
        query_intentions = query_intention_df.select(QUERY_INTENTION_COLUMN).rdd.flatMap(lambda x: x).collect()
        assert query_intentions == [DEFAULT_TOPIC_NAME] * 12

    @pytest.mark.parametrize("llm_summary_enabled", ["true", "false"])
    def test_get_query_intention(self, llm_summary_enabled):
        """Test get query intention."""
        topics_dict = {
            "Spark": ["q1", "q2", "q3"],
            "MoMo": ["q4", "q5", "q6"]
        }

        aa = json.dumps(topics_dict)
        print(aa)
        data = [
                    ["q1"], ["q2"], ["q3"], ["q4"],
                    ["q5"], ["q6"], ["q7"], ["q8"]
                ]
        data_schema = StructType(
            [
                StructField(PROMPT_COLUMN, StringType(), True)
            ]
        )
        spark = self._init_spark()
        data_df = spark.createDataFrame(data, schema=data_schema)

        query_intention_df = data_df.withColumn(QUERY_INTENTION_COLUMN, get_query_intention(col(PROMPT_COLUMN),
                                                                            lit(json.dumps(topics_dict)),
                                                                            lit(llm_summary_enabled)))
        query_intentions = query_intention_df.collect()
        query_intention_0 = query_intentions[0]
        query_intention_3 = query_intentions[3]
        query_intention_6 = query_intentions[6]
        if llm_summary_enabled == "true":
            assert query_intention_0[QUERY_INTENTION_COLUMN] == "Spark"
            assert query_intention_3[QUERY_INTENTION_COLUMN] == "MoMo"
            assert query_intention_6[QUERY_INTENTION_COLUMN] == DEFAULT_TOPIC_NAME
        else:
            assert query_intention_0[QUERY_INTENTION_COLUMN] == "topic_0"
            assert query_intention_3[QUERY_INTENTION_COLUMN] == "topic_1"
            assert query_intention_6[QUERY_INTENTION_COLUMN] == DEFAULT_TOPIC_NAME

            
