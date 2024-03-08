# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for token statistics compute metrics component."""

import json
from pyspark.sql import DataFrame
from pyspark.sql.types import StructType, StringType, DoubleType, StructField
from model_data_collector_preprocessor.span_tree_utils import SpanTreeNode
from shared_utilities.constants import (
    SIGNAL_METRICS_METRIC_NAME,
    SIGNAL_METRICS_METRIC_VALUE,
    SIGNAL_METRICS_THRESHOLD_VALUE,
)
from shared_utilities.io_utils import init_spark


class MetricsProcessor:
    """Metrics Preprocessor consists of all the function used to generate metrics for token stats."""
    def __init__(self):
        """

        Initializes all the metrics that needs calculation.
        """
        self.total_prompt_count = 0
        self.total_completion_count = 0
        self.total_token_count = 0
        self.avg_prompt_count = 0
        self.avg_completion_count = 0
        self.avg_total_count = 0
        self.model_data = {}
        self.request_data = []
        self.total_requests = 0

    def process(self, df_traces: DataFrame):
        """
        Args:
            df_traces (DataFrame): input aggregrated traces from genai processor.
        """
        df_traces = df_traces.select("root_span")
        all_spans = df_traces.toJSON().collect()
        root_input = ""
        root_output = ""
        for span_json in all_spans:
            root_span_dict = json.loads(span_json)["root_span"]
            node = SpanTreeNode.create_node_from_json_str(root_span_dict)
            if node.span_row:
                attributes = json.loads(node.span_row.asDict()["attributes"])
                root_input = attributes.get("inputs")
                root_output = attributes.get("output")

            self.process_node(root_span_dict, root_input, root_output)
        self.calculate_averages()
        print(self.model_data)
        print(self.request_data)

    def metrics_generator(self):
        """

        Returns:
        metrics_data_df (DataFrame): metrics dataframe.
        """
        spark = init_spark()
        schema = StructType([
            StructField(SIGNAL_METRICS_METRIC_NAME, StringType(), True),
            StructField(SIGNAL_METRICS_METRIC_VALUE, DoubleType(), True),
            StructField(SIGNAL_METRICS_THRESHOLD_VALUE, StringType(), True)
        ])
        metrics_data = [
            ("TotalTokenCount", self.total_token_count*1.0, 0.0),
            ("TotalPromptCount", self.total_prompt_count*1.0, 0.0),
            ("TotalCompletionCount", self.total_completion_count*1.0, 0.0),
            ("AvgTokenCount", self.avg_total_count*1.0, 0.0),
            ("AvgPromptCount", self.avg_prompt_count*1.0, 0.0),
            ("AvgCompletionCount", self.avg_completion_count*1.0, 0.0)
        ]
        for model_name, model_info in self.model_data.items():
            if model_info["model_completion_count"] is not None:
                metrics_data.append((f"{model_name}_CompletionCount", model_info["model_completion_count"]*1.0, "0.0"))
            metrics_data.append((f"{model_name}_TotalRequests", model_info["total_requests"]*1.0, "0.0"))
            metrics_data.append((f"{model_name}_PromptCount", model_info["model_prompt_count"]*1.0, "0.0"))
        metrics_data_df = spark.createDataFrame(metrics_data, schema=schema)
        print("metrics calculated")
        metrics_data_df.show()
        return metrics_data_df

    def get_latest_run(self):
        """

        Returns:
        sample_data_df (DataFrame): returns input, output and tokens for current run.
        """
        spark = init_spark()
        schema = StructType([
            StructField("PromptCount", DoubleType(), True),
            StructField("Input", StringType(), True),
            StructField("CompletionCount", DoubleType(), True),
            StructField("Output", StringType(), True)
        ])
        request_data_tuples = [(d["prompt_count"]*1.0,
                                d["input"],
                                d["completion_count"]*1.0,
                                d["output"]) for d in self.request_data]
        sample_data_df = spark.createDataFrame(request_data_tuples, schema=schema)
        sample_data_df.show()
        return sample_data_df

    def process_node(self, node_json_str: str, root_input: str, root_output: str):
        """Recursive function to iterate over children"""
        node = SpanTreeNode.create_node_from_json_str(node_json_str)
        if node.span_row:
            attributes = json.loads(node.span_row.asDict()["attributes"])
            span_type = attributes.get("span_type", "Unknown")
            if span_type in ["Embedding", "LLM"]:
                self.handle_span_type(attributes, span_type, root_input, root_output)
        for child_json_str in node.children:
            self.process_node(child_json_str, root_input, root_output)

    def handle_span_type(self, attributes: dict, span_type: str, root_input: str, root_output: str):
        """

        Args:
            attributes (dict): attributes.
            span_type (str): LLM, Embedding, Tool, Flow.
            root_input (str): input from Flow.
            root_output (str): output from Flow.
        """
        completion_count = attributes.get("llm.token_count.completion", 0)
        prompt_count = attributes.get("llm.token_count.prompt", 0)
        total_count = attributes.get("llm.token_count.total", 0)
        model = attributes.get("llm.model", "")

        self.total_prompt_count += prompt_count
        self.total_completion_count += completion_count
        self.total_token_count += total_count
        self.total_requests += 1

        if model not in self.model_data:
            self.model_data[model] = {
                "model_name": model,
                "model_type": span_type,
                "model_prompt_count": 0,
                "model_completion_count": 0 if span_type == "LLM" else None,
                "total_requests": 0
            }

        if span_type == "LLM":
            self.model_data[model]["model_completion_count"] += completion_count
        self.model_data[model]["model_prompt_count"] += prompt_count
        self.model_data[model]["total_requests"] += 1

        self.request_data.append({
            "prompt_count": prompt_count,
            "input": root_input,
            "completion_count": completion_count,
            "output": root_output
        })

    def calculate_averages(self):
        """ Calculate the averages per request"""
        self.avg_prompt_count = self.total_prompt_count / self.total_requests if self.total_requests else 0
        self.avg_completion_count = self.total_completion_count / self.total_requests if self.total_requests else 0
        self.avg_total_count = self.total_token_count / self.total_requests if self.total_requests else 0
