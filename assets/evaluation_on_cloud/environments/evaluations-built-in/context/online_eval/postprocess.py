# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Postprocess script for the online evaluation context."""
from argparse import ArgumentParser

import pandas as pd

from time import time_ns
import json
import opentelemetry
from opentelemetry import _logs
from opentelemetry.trace.span import TraceFlags
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from azure.monitor.opentelemetry.exporter import AzureMonitorLogExporter

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DEFAULT_TRACE_ID_COLUMN = "operation_Id"
DEFAULT_SPAN_ID_COLUMN = "operation_ParentId"


def get_args():
    """Get arguments from the command line."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--preprocessed_data", type=str, dest="preprocessed_data",
                        default="./preprocessed_data_output.jsonl")
    parser.add_argument("--evaluated_data", type=str, dest="evaluated_data", default="./evaluated_data_output.jsonl")
    parser.add_argument("--connection_string", type=str, dest="connection_string", default=None)
    parser.add_argument("--sampling_rate", type=str, dest="sampling_rate", default="1")
    parser.add_argument("--service_name", type=str, dest="service_name", default="evaluation.app")

    args, _ = parser.parse_known_args()
    return vars(args)


def configure_logging(args) -> LoggerProvider:
    """Configure logging."""
    logger.info("Configuring logging")
    provider = LoggerProvider()
    _logs.set_logger_provider(provider)
    args["connection_string"] = None if args["connection_string"] == "" else args["connection_string"]
    provider.add_log_record_processor(
        BatchLogRecordProcessor(AzureMonitorLogExporter(connection_string=args["connection_string"])))
    logger.info("Logging configured")
    return provider


def log_evaluation_event_single(trace_id, span_id, trace_flags, response_id, evaluation, service_name):
    """Log evaluation event."""
    for name, value in evaluation.items():
        attributes = {"event.name": f"gen_ai.evaluation.{name}", "gen_ai.evaluation.score": json.dumps(value),
                      "gen_ai.response_id": response_id, "service.name": service_name}
        body = f"gen_ai.evaluation for response_id: {response_id}"

        event = opentelemetry.sdk._logs.LogRecord(
            timestamp=time_ns(),
            observed_timestamp=time_ns(),
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=trace_flags,
            severity_text=None,
            severity_number=_logs.SeverityNumber.UNSPECIFIED,
            body=body,
            attributes=attributes
        )
        _logs.get_logger(__name__).emit(event)


def log_evaluation_event(row) -> None:
    """Log evaluation event."""
    if "trace_id" not in row or "span_id" not in row or "evaluation" not in row:
        logger.warning("Missing required fields in the row: trace_id, span_id, evaluation")

    trace_id = int(row.get("trace_id", row.get(DEFAULT_TRACE_ID_COLUMN, "0")), 16)
    span_id = int(row.get("span_id", row.get(DEFAULT_SPAN_ID_COLUMN, "0")), 16)
    trace_flags = TraceFlags(TraceFlags.SAMPLED)
    response_id = row.get("gen_ai_response_id", "")
    evaluation_results = row.get("evaluation", {})
    service_name = row.get("service_name", "evaluation.app")
    if isinstance(evaluation_results, dict):
        evaluation_results = [evaluation_results]
    for evaluation in evaluation_results:
        log_evaluation_event_single(trace_id, span_id, trace_flags, response_id, evaluation, service_name)


def get_combined_data(preprocessed_data, evaluated_data, service_name):
    """Combine preprocessed and evaluated data."""
    logger.info("Combining preprocessed and evaluated data.")
    preprocessed_df = pd.read_json(preprocessed_data, lines=True)
    evaluation_data = []
    with open(evaluated_data, 'r') as file:
        for line in file:
            evaluation_data.append(json.loads(line))

    preprocessed_df["evaluation"] = evaluation_data
    preprocessed_df["service_name"] = service_name
    return preprocessed_df


def run(args):
    """Entry point of model prediction script."""
    logger.info(f"Commandline args:> Service Name: {args['service_name']}")
    provider = configure_logging(args)
    data = get_combined_data(args["preprocessed_data"], args["evaluated_data"],
                             args["service_name"])
    for _, row in data.iterrows():
        log_evaluation_event(row)
    provider.force_flush()


if __name__ == "__main__":
    args = get_args()
    run(args)
