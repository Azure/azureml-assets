# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for token statistics compute metrics component."""

import json
from genai_token_statistics_compute_metrics.constants import (
    INCLUDE_SPAN_TYPE,
    COMPLETION_COUNT_KEYS,
    PROMPT_COUNT_KEYS,
    TOTAL_COUNT_KEYS,
    MODEL_KEYS)
from pyspark.sql.functions import sum as _sum, avg as _avg, count as _count, lit
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StringType, StructField, IntegerType
from shared_utilities.span_tree_utils import SpanTreeNode, SpanTree
from shared_utilities.constants import (
    ROOT_SPAN_COLUMN
)
from typing import Optional


class MetricsProcessor:
    """Processor for generating token statistics metrics."""

    def __init__(self):
        """Initialize metrics for computation."""
        self.metrics_df = []
        self.requests_df = []
        self.metrics_model_df = []

    def create_span_tree_from_dataframe_row(self, row: Row) -> Optional[SpanTreeNode]:
        """Create a SpanTree from a DataFrame row."""
        if row.root_span is None:
            return None
        return SpanTree.create_tree_from_json_string(row.root_span)

    def process(self, df_traces: DataFrame):
        """Process the aggregated trace data to compute metrics.

        Args:
            df_traces (DataFrame): Aggregated traces from the GenAI processor.
        """
        df_traces = df_traces.select(ROOT_SPAN_COLUMN)
        # Create an RDD of tuples with the necessary counts
        counts_and_requests_rdd = df_traces.rdd.flatMap(self.process_row)
        counts_rdd = counts_and_requests_rdd.filter(lambda x: isinstance(x, tuple))
        requests_rdd = counts_and_requests_rdd.filter(lambda x: isinstance(x, dict))

        # Convert the RDD to a DataFrame
        counts_schema = StructType([
            StructField("prompt_count", IntegerType(), True),
            StructField("completion_count", IntegerType(), True),
            StructField("total_count", IntegerType(), True),
            StructField("model", StringType(), True)
        ])
        counts_df = counts_rdd.toDF(schema=counts_schema)

        requests_schema = StructType([
            StructField("prompt_count", IntegerType(), True),
            StructField("input", StringType(), True),
            StructField("completion_count", IntegerType(), True),
            StructField("output", StringType(), True),
        ])
        requests_df = requests_rdd.toDF(schema=requests_schema)

        # Perform aggregation to compute the metrics
        metrics_df = counts_df.agg(
            _sum('prompt_count').alias('total_prompt_count'),
            _sum('completion_count').alias('total_completion_count'),
            _sum('total_count').alias('total_token_count'),
            _avg('prompt_count').alias('avg_prompt_count'),
            _avg('completion_count').alias('avg_completion_count'),
            _avg('total_count').alias('avg_total_count'),
            _count(lit(1)).alias('total_requests')
        )
        metrics_model_df = counts_df.groupBy("model").agg(
            _sum('prompt_count').alias('total_prompt_count'),
            _sum('completion_count').alias('total_completion_count'),
            _count(lit(1)).alias('total_requests')
        )

        # Store the metrics DataFrame as an instance variable for later use
        self.metrics_model_df = metrics_model_df
        self.metrics_df = metrics_df
        self.requests_df = requests_df

    def process_row(self, row: Row):
        """Process the aggregated trace data to compute metrics.

        Args:
            df_traces (DataFrame): Aggregated traces from the GenAI processor.
        """
        tree = self.create_span_tree_from_dataframe_row(row)

        if tree is None:
            return []
        counts_and_requests = []
        for span in tree:
            span_type = span.span_type
            if span_type in INCLUDE_SPAN_TYPE:
                attributes = json.loads(span.get_node_attribute(attribute_key="attributes"))
                # in some cases we have LLM/Embedding span with no parent so need to check
                # before accessing its parent object
                parent = tree.get_span_tree_node_by_span_id(span.parent_id)
                parent_input = None
                parent_output = None
                if parent is not None:
                    parent_input = parent.input
                    parent_output = parent.output
                # Extract the counts and model information
                completion_count = self.get_value_from_attributes(attributes, COMPLETION_COUNT_KEYS)
                prompt_count = self.get_value_from_attributes(attributes, PROMPT_COUNT_KEYS)
                total_count = self.get_value_from_attributes(attributes, TOTAL_COUNT_KEYS)
                model = self.get_value_from_attributes(attributes, MODEL_KEYS)

                # Append a tuple with the counts and model to the list
                if model is not None:
                    counts_and_requests.append((prompt_count, completion_count, total_count, model))

                # Append requests data if available
                if parent_input is not None and parent_output is not None:
                    counts_and_requests.append({
                        "prompt_count": prompt_count,
                        "input": parent_input,
                        "completion_count": completion_count,
                        "output": parent_output
                    })

        # Return the list of tuples
        return counts_and_requests

    def get_value_from_attributes(self, attributes, keys):
        """Get usage values from attributes."""
        for key in keys:
            if key in attributes:
                return attributes.get(key)
        return 0

    def has_completion_count(self, span_type):
        """Check if model is completion type."""
        return span_type == "LLM"
