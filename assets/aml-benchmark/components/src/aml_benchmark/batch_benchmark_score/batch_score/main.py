# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The entry script for batch benchmark inference."""

import traceback
import os
import json
import asyncio
import sys
import pandas as pd
from argparse import ArgumentParser, Namespace

from azureml._common._error_definition.azureml_error import AzureMLError

from aml_benchmark.utils.online_endpoint.endpoint_utils import EndpointUtilities
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel

from .utils.exceptions import BenchmarkValidationException
from .utils.error_definitions import BenchmarkValidationError

from .utils.token_provider import TokenProvider
from .utils.tally_failed_request_handler import TallyFailedRequestHandler
from . import sequential
from . import parallel_driver as parallel
from .utils.common import constants
from .utils.trace_configs import (
    ExceptionTrace,
    ResponseChunkReceivedTrace,
    RequestEndTrace,
    RequestRedirectTrace,
    ConnectionReuseconnTrace,
    ConnectionCreateStartTrace,
    ConnectionCreateEndTrace
)
from .utils.scoring_client import ScoringClient
from .utils.common.common import str2bool, convert_to_list
from .utils import logging_utils as lu
from .utils.logging_utils import setup_logger, get_logger, get_events_client, set_mini_batch_id
from .utils.common.json_encoder_extensions import setup_encoder
from .header_handlers.header_handler import HeaderHandler
from .header_handlers.oss.oss_header_handler import OSSHeaderHandler
from .header_handlers.oai.oai_header_handler import OAIHeaderHandler


par: parallel.Parallel = None
seq: sequential.Sequential = None
args: Namespace = None


def init():
    """PRS init method."""
    global par
    global seq
    global args

    print("Entered init")

    parser = ArgumentParser()

    setup_arguments(parser)
    args, unknown_args = parser.parse_known_args()
    setup_logger("DEBUG" if args.debug_mode else "INFO", args.app_insights_connection_string)
    logger = get_logger()

    setup_encoder(args.ensure_ascii)

    print_arguments(args)

    scoring_client = setup_scoring_client()
    trace_configs = setup_trace_configs()
    request_input_transformer = None
    logging_input_transformer = None

    if args.run_type == "sequential":
        logger.info("using sequential run type")

        seq = sequential.Sequential(
            scoring_client=scoring_client,
            segment_large_requests=args.segment_large_requests,
            segment_max_token_size=args.segment_max_token_size,
            trace_configs=trace_configs,
            request_input_transformer=request_input_transformer,
            logging_input_transformer=logging_input_transformer)

    if args.run_type == "parallel":
        logger.info("using parallel run type")

        loop = setup_loop()
        conductor = parallel.Conductor(
            loop=loop,
            scoring_client=scoring_client,
            segment_large_requests=args.segment_large_requests,
            segment_max_token_size=args.segment_max_token_size,
            initial_worker_count=args.initial_worker_count,
            max_worker_count=args.max_worker_count,
            trace_configs=trace_configs,
            max_retry_time_interval=args.max_retry_time_interval)

        par = parallel.Parallel(
            loop=loop,
            conductor=conductor,
            request_input_transformer=request_input_transformer,
            logging_input_transformer=logging_input_transformer)

    get_events_client().emit_batch_driver_init(job_params=vars(args))


def run(input_data: pd.DataFrame, mini_batch_context):
    """PRS run method."""
    global par
    global seq
    global args

    lu.get_logger().info("Running new mini-batch...")
    set_mini_batch_id(mini_batch_context.minibatch_index)

    ret = []

    data_list = convert_to_list(input_data, additional_properties=args.additional_properties)

    get_events_client().emit_mini_batch_started(input_row_count=len(data_list))
    lu.get_logger().info("Number of payloads in this mini_batch: {}".format(len(data_list)))

    try:
        if args.run_type == "sequential":
            ret = seq.start(data_list)
        elif args.run_type == "parallel":
            ret = par.start(data_list)
        else:
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details="Invalid run type")
            )

        if args.save_mini_batch_results == "enabled":
            lu.get_logger().info("save_mini_batch_results is enabled")
            save_mini_batch_results(ret, mini_batch_context)
        else:
            lu.get_logger().info("save_mini_batch_results is disabled")

    except Exception as e:
        get_events_client().emit_mini_batch_completed(
            input_row_count=len(data_list),
            output_row_count=len(ret),
            exception=type(e).__name__,
            stacktrace=traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        )
        raise e
    else:
        get_events_client().emit_mini_batch_completed(
            input_row_count=len(data_list), output_row_count=len(ret)
        )
    finally:
        lu.get_logger().info("Completed mini-batch.")
        set_mini_batch_id(None)

    return get_return_value(ret, args.output_behavior)


def shutdown():
    """Shutdown method."""
    global par
    global seq
    global args

    lu.get_logger().info("Starting shutdown")
    get_events_client().emit_batch_driver_shutdown(job_params=vars(args))

    if args.run_type == "parallel":
        par.close_session()


def setup_arguments(parser: ArgumentParser):
    """Add run arguments to parser."""
    # Local runs only
    parser.add_argument("--token_file_path", default=None, type=str)

    # Component parameters
    parser.add_argument("--debug_mode", default=False, type=str2bool)
    parser.add_argument("--app_insights_connection_string", nargs='?', const=None, type=str)
    parser.add_argument("--additional_properties", nargs='?', const=None, type=str)
    parser.add_argument("--run_type", type=str)
    parser.add_argument("--model_type", type=str)

    # Sequential-specific parameters
    parser.add_argument("--sorted", default=False, type=str2bool)
    parser.add_argument("--max_prompts", default=2500, type=int)

    # Parallel-specific parameters
    parser.add_argument("--initial_worker_count", default=5, type=int)
    parser.add_argument("--max_worker_count", default=200, type=int)

    parser.add_argument("--online_endpoint_url", type=str)
    parser.add_argument("--max_retry_time_interval", nargs='?', const=None, default=None, type=int)
    parser.add_argument("--segment_large_requests", default="disabled", type=str)
    parser.add_argument("--segment_max_token_size", default=800, type=int)
    parser.add_argument("--save_mini_batch_results", type=str)
    parser.add_argument("--mini_batch_results_out_directory", type=str)
    parser.add_argument("--ensure_ascii", default=False, type=str2bool)
    parser.add_argument("--output_behavior", type=str)
    parser.add_argument("--batch_pool", nargs='?', const=None, default=None, type=str)
    parser.add_argument("--request_path", default='score', type=str)
    parser.add_argument("--service_namespace")
    parser.add_argument("--quota_audience")
    parser.add_argument("--quota_estimator", default="completion")
    parser.add_argument("--user_agent_segment", nargs='?', const=None, default=None, type=str)
    parser.add_argument("--additional_headers", nargs='?', const=None, type=str)
    parser.add_argument("--tally_failed_requests", default=False, type=str2bool)
    parser.add_argument("--tally_exclusions", default="none", type=str)

    parser.add_argument("--image_input_folder", nargs='?', const=None, type=str)
    parser.add_argument("--endpoint_subscription_id", default=None, type=str)
    parser.add_argument("--endpoint_resource_group", default=None, type=str)
    parser.add_argument("--endpoint_workspace", default=None, type=str)
    parser.add_argument("--input_metadata", default=None, type=str)
    parser.add_argument("--deployment_name", default=None, type=str)
    parser.add_argument("--connections_name", default=None, type=str)


def print_arguments(args: Namespace):
    """Print all input arguments."""
    lu.get_logger().debug("token_file_path path: %s" % args.token_file_path)

    lu.get_logger().debug("debug_mode path: %s" % args.debug_mode)
    lu.get_logger().debug(
        "app_insights_connection_string: %s" % "None" if args.app_insights_connection_string is None else "[redacted]"
    )
    lu.get_logger().debug("additional_properties path: %s" % args.additional_properties)
    lu.get_logger().debug("run_type path: %s" % args.run_type)

    lu.get_logger().debug("sorted path: %s" % args.sorted)
    lu.get_logger().debug("max_prompts path: %s" % args.max_prompts)

    lu.get_logger().debug("initial_worker_count path: %s" % args.initial_worker_count)
    lu.get_logger().debug("max_worker_count path: %s" % args.max_worker_count)

    lu.get_logger().debug("max_retry_time_interval path: %s" % args.max_retry_time_interval)
    lu.get_logger().debug("segment_large_requests path: %s" % args.segment_large_requests)
    lu.get_logger().debug("segment_max_token_size path: %s" % args.segment_max_token_size)
    lu.get_logger().debug("save_mini_batch_results: %s" % args.save_mini_batch_results)
    lu.get_logger().debug("mini_batch_results_out_directory: %s" % args.mini_batch_results_out_directory)
    lu.get_logger().debug("ensure_ascii path: %s" % args.ensure_ascii)
    lu.get_logger().debug("output_behavior: %s" % args.output_behavior)
    lu.get_logger().debug("batch_pool: %s" % args.batch_pool)
    lu.get_logger().debug("request_path: %s" % args.request_path)
    lu.get_logger().debug(f"service_namespace: {args.service_namespace}")
    lu.get_logger().debug(f"quota_audience: {args.quota_audience}")
    lu.get_logger().debug(f"quota_estimator: {args.quota_estimator}")
    lu.get_logger().debug(f"user_agent_segment: {args.user_agent_segment}")
    lu.get_logger().debug("online_endpoint_url: %s" % args.online_endpoint_url)
    lu.get_logger().debug("additional_headers: %s" % args.additional_headers)
    lu.get_logger().debug("tally_failed_requests: %s" % args.tally_failed_requests)
    lu.get_logger().debug("tally_exclusions: %s" % args.tally_exclusions)

    lu.get_logger().debug("image_input_folder: %s" % args.image_input_folder)
    lu.get_logger().debug("endpoint_subscription_id: %s" % args.endpoint_subscription_id)
    lu.get_logger().debug("endpoint_resource_group: %s" % args.endpoint_resource_group)
    lu.get_logger().debug("endpoint_workspace: %s" % args.endpoint_workspace)
    lu.get_logger().debug("mdoel_type: %s" % args.model_type)
    lu.get_logger().debug("deployment_name: %s" % args.deployment_name)
    lu.get_logger().debug("connections_name: %s" % args.connections_name)


def save_mini_batch_results(mini_batch_results: list, mini_batch_context):
    """Save mini batch results."""
    mini_batch_results_out_directory = args.mini_batch_results_out_directory
    lu.get_logger().info("mini_batch_results_out_directory: {}".format(mini_batch_results_out_directory))

    filename = f"{mini_batch_context.minibatch_index}.jsonl"
    file_path = os.path.join(mini_batch_results_out_directory, filename)

    lu.get_logger().info(f"Start saving {len(mini_batch_results)} results to file {file_path}")
    with open(file_path, "w", encoding="utf-8") as writer:
        for item in mini_batch_results:
            writer.write(item + "\n")
    lu.get_logger().info(f"Completed saving {len(mini_batch_results)} results to file {file_path}")


def setup_trace_configs():
    """Init trace configs."""
    is_enabled = os.environ.get(constants.BATCH_SCORE_TRACE_LOGGING, None)
    trace_configs = None

    if is_enabled and is_enabled.lower() == "true":
        lu.get_logger().info("Trace logging enabled, populating trace_configs.")
        trace_configs = [
            ExceptionTrace(), ResponseChunkReceivedTrace(), RequestEndTrace(), RequestRedirectTrace(),
            ConnectionCreateStartTrace(), ConnectionCreateEndTrace(), ConnectionReuseconnTrace()]
    else:
        lu.get_logger().info("Trace logging disabled")

    return trace_configs


def get_return_value(ret: 'list[str]', output_behavior: str):
    """Get return value."""
    if (output_behavior == "summary_only"):
        lu.get_logger().info("Returning results in summary_only mode")
        # PRS confirmed there is no way to allow users to toggle the output_action behavior in the v2 component.
        # A workaround is to return an empty array, but that can run afoul of the item-based error_threshold logic.
        # Instead, return an array of the same length as the results array, but with dummy values.
        # This is what core-search did in their fork of the component.
        return ["True"] * len(ret)

    lu.get_logger().info("Returning results in append_row mode")
    return ret


def setup_loop() -> asyncio.AbstractEventLoop:
    """Start event loop."""
    if sys.platform == 'win32':
        # For windows environment
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    return asyncio.new_event_loop()


def setup_scoring_client() -> ScoringClient:
    """Start scoring client."""
    token_provider = TokenProvider(
        token_file_path=args.token_file_path
    )

    tally_handler = TallyFailedRequestHandler(
        enabled=args.tally_failed_requests,
        tally_exclusions=args.tally_exclusions
    )

    input_metadata = {}
    if args.input_metadata is not None:
        input_metadata = EndpointUtilities.load_endpoint_metadata_json(args.input_metadata)

    online_endpoint_url = input_metadata.get("scoring_url", args.online_endpoint_url)
    headers = input_metadata.get("scoring_headers", {})
    input_additional_headers = args.additional_headers
    if input_additional_headers:
        headers.update(json.loads(input_additional_headers))

    header_handler = setup_header_handler(
        token_provider=token_provider, model_type=args.model_type, input_metadata=input_metadata,
        input_headers=json.dumps(headers), scoring_url=online_endpoint_url
    )

    scoring_client = ScoringClient(
        header_handler=header_handler,
        quota_client=None,
        routing_client=None,
        online_endpoint_url=online_endpoint_url,
        tally_handler=tally_handler,
    )

    return scoring_client


def setup_header_handler(
        token_provider: TokenProvider, model_type: str, input_metadata: dict, input_headers: str,
        scoring_url: str
) -> HeaderHandler:
    """Add header handler."""
    endpoint_workspace = input_metadata.get("workspace_name", args.endpoint_workspace)
    endpoint_resource_group = input_metadata.get("resource_group", args.endpoint_resource_group)
    endpoint_subscription_id = input_metadata.get("subscription_id", args.endpoint_subscription_id)
    endpoint_name = input_metadata.get("endpoint_name", scoring_url.split('/')[2].split('.')[0])
    deployment_name = input_metadata.get("deployment_name", args.deployment_name)
    connections_name = input_metadata.get("connection_name", args.connections_name)

    model = OnlineEndpointModel(
        model=None, model_version=None, model_type=model_type, endpoint_url=scoring_url)
    if model.is_aoai_model():
        return OAIHeaderHandler(
            token_provider=token_provider, user_agent_segment=args.user_agent_segment,
            batch_pool=args.batch_pool,
            quota_audience=args.quota_audience,
            additional_headers=input_headers,
            endpoint_name=endpoint_name,
            endpoint_subscription=endpoint_subscription_id,
            endpoint_resource_group=endpoint_resource_group,
            deployment_name=deployment_name,
            connections_name=connections_name,
            online_endpoint_model=model
        )
    return OSSHeaderHandler(
        token_provider=token_provider, user_agent_segment=args.user_agent_segment,
        batch_pool=args.batch_pool,
        quota_audience=args.quota_audience,
        additional_headers=input_headers,
        endpoint_name=endpoint_name,
        endpoint_subscription=endpoint_subscription_id,
        endpoint_resource_group=endpoint_resource_group,
        endpoint_workspace=endpoint_workspace,
        deployment_name=deployment_name,
        connections_name=connections_name,
        online_endpoint_model=model
    )
