# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preprocess script for the online evaluation context."""
import pandas as pd
from datetime import datetime, timedelta
from argparse import ArgumentParser

from azure.monitor.query import LogsQueryStatus

from utils import get_app_insights_client
from logging_utilities import get_logger
from constants import Queries

logger = get_logger(name=__name__)


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
    parser.add_argument("--schedule_name", type=str, dest="schedule_name", default=None)
    parser.add_argument("--schedule_start_time", type=str, dest="schedule_start_time", default=None)
    # parser.add_argument("--evaluators", type=str, dest="evaluators")

    parser.add_argument("--preprocessed_data", type=str, dest="preprocessed_data",
                        default="./preprocessed_data_output.jsonl")
    args, _ = parser.parse_known_args()
    return vars(args)


# Function to fetch the last run time from App Insights
def calculate_time_window(client, resource_id, schedule_id, provided_start_time):
    """Calculate the time window for a job instance based on the system's current time and previous run time."""
    query = Queries.LAST_EXECUTION_TIME_QUERY.format(schedule_id=schedule_id)

    # Define timespan for the query
    current_time_str = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
    current_time = datetime.strptime(current_time_str, '%Y-%m-%d %H:%M:%S')
    provided_start_datetime = datetime.strptime(provided_start_time, '%Y-%m-%d %H:%M:%S')
    default_timespan_start = max(provided_start_datetime, current_time - timedelta(days=30))
    timespan = (default_timespan_start, current_time)

    response = client.query_resource(resource_id, query=query, timespan=timespan)

    if response.tables:
        try:
            rows = response.tables[0].rows
            if rows:
                log_message = rows[0][0]
                # Extract the timestamp from the logged message
                if "Last Run Time:" in log_message:
                    timestamp_str = log_message.split("Last Run Time:")[1].strip()
                    return datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S'), current_time
        except Exception as e:
            logger.info(f"Error while parsing the last run time: {e}")
    return default_timespan_start, current_time


def get_logs(client, resource_id: str, query: str, start_time: datetime, end_time: datetime):
    """Get logs from the resource."""
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
        logger.info(f"Query returned {len(df)} rows, {len(df.columns)} columns, and df.columns: {df.columns}")
        return df
    except Exception as e:
        logger.info("something fatal happened")
        logger.info(e)


def save_output(result, args):
    """Save output."""
    try:
        # Todo: One conversation will be split across multiple rows. how to combine them?
        logger.info(f"Saving output to {args['preprocessed_data']}")
        result.to_json(args["preprocessed_data"], orient="records", lines=True)
    except Exception as e:
        logger.info("Unable to save output.")
        raise e


def run(args):
    """Entry point of model prediction script."""
    logger.info(
        f"Connection type: {args['preprocessor_connection_type']}, Resource ID: {args['resource_id']}, Cron "
        f"Expression: {args['cron_expression']}, Sampling Rate: {args['sampling_rate']}")
    client = get_app_insights_client(use_managed_identity=args["preprocessor_connection_type"] == "managed-identity")
    start_time, end_time = calculate_time_window(client, args["resource_id"], args["schedule_name"],
                                                 args["schedule_start_time"])
    logger.info(f"Start Time: {start_time}, End Time: {end_time}")
    result = get_logs(client, args["resource_id"], args["query"], start_time, end_time)
    save_output(result, args)
    args["time_window"] = (start_time, end_time)


if __name__ == "__main__":
    args = get_args()
    run(args)
