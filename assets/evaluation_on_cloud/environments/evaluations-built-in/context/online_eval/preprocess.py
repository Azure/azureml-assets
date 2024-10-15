# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Client for AppInsights service."""
import pandas as pd
from croniter import croniter
from datetime import datetime
from argparse import ArgumentParser
import os

from azure.monitor.query import LogsQueryStatus
from azure.core.exceptions import HttpResponseError

from utils import get_app_insights_client

# The global vaiables will be removed
from exceptions import DataSavingException
from error_definitions import SavingOutputError, NoDataFoundError
from azureml.telemetry.activity import log_activity
from logging_utilities import swallow_all_exceptions, flush_logger, get_logger, log_traceback, custom_dimensions, \
    current_run, get_azureml_exception
from azureml.automl.core.shared.logging_utilities import mark_path_as_loggable
import constants

# Mark current path as allowed
mark_path_as_loggable(os.path.dirname(__file__))
custom_dimensions.app_name = constants.TelemetryConstants.COMPONENT_NAME
logger = get_logger(name=__name__)
test_run = current_run.run
root_run = current_run.root_run
ws = current_run.workspace
custom_dims_dict = vars(custom_dimensions)


def get_args():
    """Get arguments from the command line."""
    parser = ArgumentParser()
    # Inputs
    parser.add_argument("--resource_id", type=str, dest="resource_id")
    parser.add_argument("--query", type=str, dest="query")
    parser.add_argument("--sampling_rate", type=str, dest="sampling_rate", default="1")
    parser.add_argument("--preprocessor_connection_type", type=str, dest="preprocessor_connection_type",
                        default="user-identity")
    parser.add_argument("--cron_expression", type=str, dest="cron_expression", default="0 0 * * *")
    # parser.add_argument("--evaluators", type=str, dest="evaluators")

    parser.add_argument("--preprocessed_data", type=str, dest="preprocessed_data",
                        default="./preprocessed_data_output.jsonl")
    args, _ = parser.parse_known_args()
    return vars(args)


def calculate_time_window(cron_expression: str):
    """
    Calculates the time window for a job instance based on the system's current time and its cron schedule.
    """
    # Parse the current time
    logger.info(f"Calculating time window for cron expression: {cron_expression}")
    current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    current_time = datetime.strptime(current_time_str, '%Y-%m-%d %H:%M:%S')
    # Initialize croniter based on the cron expression and current time
    cron = croniter(cron_expression, current_time)
    # Calculate the previous run time (T_prev)
    # now = datetime.now().replace(second=0, microsecond=0)
    # The below statement would return the current cron's timestamp
    _ = cron.get_prev(datetime)
    # Window would be from the previous cron's timestamp to the current cron's timestamp
    return cron.get_prev(datetime), current_time
    # return cron.get_prev(datetime), cron.get_next(datetime)


def get_logs(client, resource_id: str, query: str, start_time: datetime, end_time: datetime):
    try:
        logger.info(f"Querying resource: {resource_id}")
        response = client.query_resource(resource_id, query, timespan=(start_time, end_time))
        if response.status == LogsQueryStatus.SUCCESS:
            data = response.tables
        else:
            # LogsQueryPartialResult
            error = response.partial_error
            data = response.partial_data
            logger.info(error)
        if len(data) == 0 or len(data) > 1:
            raise Exception(f"Unable to parse query results. Unexpected number of tables: {len(data)}.")
        table = data[0]
        df = pd.DataFrame(data=table.rows, columns=table.columns)
        return df
    except HttpResponseError as err:
        logger.info("something fatal happened")
        logger.info(err)


def save_output(result, args):
    """Save output."""
    with log_activity(logger, constants.LogActivityLiterals.LOG_AND_SAVE_OUTPUT,
                      custom_dimensions=custom_dims_dict):
        try:
            logger.info("Saving output.")
            flush_logger(logger)
            # Todo: One conversation will be split across multiple rows. how to combine them?
            result.to_json(args["preprocessed_data"], orient="records", lines=True)
        except Exception as e:
            exception_type = SavingOutputError
            if result is not None and len(result) == 0:
                exception_type = NoDataFoundError
            exception = get_azureml_exception(DataSavingException, exception_type, e, error=repr(e))
            log_traceback(exception, logger)
            raise exception


@swallow_all_exceptions(logger)
def run(args):
    """Entry point of model prediction script."""

    logger.info(
        f"Connection type: {args['preprocessor_connection_type']}, Resource ID: {args['resource_id']}, Cron Expression: {args['cron_expression']}, Sampling Rate: {args['sampling_rate']}")
    client = get_app_insights_client(use_managed_identity=args["preprocessor_connection_type"] == "managed-identity")
    start_time, end_time = calculate_time_window(args["cron_expression"])
    logger.info(f"Start Time: {start_time}, End Time: {end_time}")
    result = get_logs(client, args["resource_id"], args["query"], start_time, end_time)
    save_output(result, args)


if __name__ == "__main__":
    args = get_args()
    run(args)
