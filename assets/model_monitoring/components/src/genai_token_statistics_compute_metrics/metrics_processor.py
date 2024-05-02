# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for token statistics compute metrics component."""

import json
from genai_token_statistics_compute_metrics.constants import (
    INCLUDE_SPAN_TYPE,
    TOTAL_TOKEN_COUNT,
    TOTAL_PROMPT_COUNT,
    TOTAL_COMPLETION_COUNT,
    AVG_TOKEN_COUNT,
    AVG_PROMPT_COUNT,
    AVG_COMPLETION_COUNT,
    MODEL_COMPLETION_COUNT,
    COMPLETION_COUNT_KEYS,
    PROMPT_COUNT_KEYS,
    TOTAL_COUNT_KEYS,
    MODEL_KEYS)
from pyspark.sql import DataFrame, Row
from pyspark.sql.types import StructType, StringType, DoubleType, StructField
from shared_utilities.span_tree_utils import SpanTreeNode, SpanTree
from shared_utilities.constants import (
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE,
    ROOT_SPAN_COLUMN
)
from shared_utilities.io_utils import init_spark
from typing import Optional, List


class MetricsProcessor:
    """Processor for generating token statistics metrics."""

    def __init__(self):
        """Initialize metrics for computation."""
        self.total_prompt_count = 0
        self.total_completion_count = 0
        self.total_token_count = 0
        self.avg_prompt_count = 0
        self.avg_completion_count = 0
        self.avg_total_count = 0
        self.model_data = {}
        self.request_data = []
        self.total_requests = 0

    def create_span_tree_from_dataframe_row(self, row: Row) -> Optional[SpanTreeNode]:
        """Create a SpanTree from a DataFrame row."""
        if row.root_span is None:
            return None
        return SpanTree.create_tree_from_json_string(row.root_span)

    def create_span_tree_from_dataframe(self, df: DataFrame) -> List[Optional[SpanTreeNode]]:
        """Create a list of SpanTrees from a DataFrame."""
        # Convert the DataFrame to an RDD and then to a list of Rows
        rows = df.rdd.map(lambda row: row).collect()

        # Create a SpanTree for each row
        span_trees = [self.create_span_tree_from_dataframe_row(row) for row in rows]
        return span_trees

    def process(self, df_traces: DataFrame):
        """Process the aggregated trace data to compute metrics.

        Args:
            df_traces (DataFrame): Aggregated traces from the GenAI processor.
        """
        df_traces = df_traces.select(ROOT_SPAN_COLUMN)
        span_trees = self.create_span_tree_from_dataframe(df_traces)

        for tree in span_trees:
            for span in tree:
                span_type = span.span_type
                if span_type in INCLUDE_SPAN_TYPE:
                    attributes = json.loads(span.get_node_attribute(attribute_key="attributes"))
                    parent = tree.get_span_tree_node_by_span_id(span.parent_id)
                    # in some cases we have LLM/Embedding span with no parent so need to check before accessing its parent object
                    parent_input = None
                    parent_output = None
                    if parent is not None:
                        parent_input = parent.input
                        parent_output = parent.output

                    self.calculate_metrics(attributes=attributes,
                                           span_type=span_type,
                                           root_input=parent_input,
                                           root_output=parent_output)
        self.calculate_averages()

    def metrics_generator(self):
        """Return: metrics_data_df (DataFrame): metrics dataframe."""
        spark = init_spark()
        schema = StructType([
            StructField(SIGNAL_METRICS_METRIC_NAME, StringType(), True),
            StructField(SIGNAL_METRICS_METRIC_VALUE, DoubleType(), True),
            StructField(SIGNAL_METRICS_THRESHOLD_VALUE, StringType(), True)
        ])
        metrics_data = [
            (TOTAL_TOKEN_COUNT, float(self.total_token_count), 0.0),
            (TOTAL_PROMPT_COUNT, float(self.total_prompt_count), 0.0),
            (TOTAL_COMPLETION_COUNT, float(self.total_completion_count), 0.0),
            (AVG_TOKEN_COUNT, float(self.avg_total_count), 0.0),
            (AVG_PROMPT_COUNT, float(self.avg_prompt_count), 0.0),
            (AVG_COMPLETION_COUNT, float(self.avg_completion_count), 0.0)
        ]
        for model_name, model_info in self.model_data.items():
            if MODEL_COMPLETION_COUNT in model_info:
                metrics_data.append((f"{model_name}.CompletionCount",
                                     float(model_info[MODEL_COMPLETION_COUNT]),
                                     0.0))
            metrics_data.append((f"{model_name}.TotalRequests", float(model_info["total_requests"]), 0.0))
            metrics_data.append((f"{model_name}.PromptCount", float(model_info["model_prompt_count"]), 0.0))
        metrics_data_df = spark.createDataFrame(metrics_data, schema=schema)
        print("metrics calculated")
        metrics_data_df.show()
        return metrics_data_df

    def get_latest_run(self):
        """Return: sample_data_df (DataFrame): returns input, output and tokens for current run."""
        spark = init_spark()
        schema = StructType([
            StructField("PromptCount", DoubleType(), True),
            StructField("Input", StringType(), True),
            StructField("CompletionCount", DoubleType(), True),
            StructField("Output", StringType(), True)
        ])
        request_data_tuples = [(float(d["prompt_count"]),
                                d["input"],
                                float(d["completion_count"]),
                                d["output"]) for d in self.request_data]
        sample_data_df = spark.createDataFrame(request_data_tuples, schema=schema)
        sample_data_df.show()
        return sample_data_df

    def calculate_metrics(self, attributes: dict, span_type: str, root_input: Optional[str], root_output: Optional[str]):
        """Args.

        attributes (dict): attributes.
        span_type (str): LLM, Embedding, Tool, Flow.
        root_input Optional(str): input from Flow.
        root_output Optional(str): output from Flow.
        """
        completion_count = self.get_value_from_attributes(attributes, COMPLETION_COUNT_KEYS)
        prompt_count = self.get_value_from_attributes(attributes, PROMPT_COUNT_KEYS)
        total_count = self.get_value_from_attributes(attributes, TOTAL_COUNT_KEYS)
        model = self.get_value_from_attributes(attributes, MODEL_KEYS)
        if model is not None:
            # TODO:Convert this calculation to pyspark logic.
            self.total_prompt_count += prompt_count
            self.total_completion_count += completion_count
            self.total_token_count += total_count
            self.total_requests += 1

            if model not in self.model_data:
                self.model_data[model] = {
                    "model_name": model,
                    "model_type": span_type,
                    "model_prompt_count": 0,
                    "model_completion_count": 0,
                    "total_requests": 0
                }
            if self.has_completion_count(span_type):
                self.model_data[model]["model_completion_count"] += completion_count
            self.model_data[model]["model_prompt_count"] += prompt_count
            self.model_data[model]["total_requests"] += 1

        if root_input is not None and root_output is not None:

            self.request_data.append({
                "prompt_count": prompt_count,
                "input": root_input,
                "completion_count": completion_count,
                "output": root_output
            })

    def get_value_from_attributes(self, attributes, keys):
        """Get usage values from attributes."""
        for key in keys:
            if key in attributes:
                return attributes.get(key)
        return 0

    def has_completion_count(self, span_type):
        """Check if model is completion type."""
        return span_type == "LLM"

    def calculate_averages(self):
        """Calculate the averages per request."""
        self.avg_prompt_count = self.total_prompt_count / self.total_requests if self.total_requests else 0
        self.avg_completion_count = self.total_completion_count / self.total_requests if self.total_requests else 0
        self.avg_total_count = self.total_token_count / self.total_requests if self.total_requests else 0
