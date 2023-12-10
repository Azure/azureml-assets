# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for batch score component."""

import asyncio
import os
import sys
import traceback

import pandas as pd

from .aoai.scoring.aoai_scoring_client import AoaiScoringClient
from .batch_pool.quota.quota_client import QuotaClient
from .batch_pool.routing.routing_client import RoutingClient
from .batch_pool.scoring.scoring_client import ScoringClient
from .common import constants
from .common.auth.token_provider import TokenProvider
from .common.configuration.configuration import Configuration
from .common.configuration.configuration_parser import ConfigurationParser
from .common.parallel import parallel_driver as parallel
from .common.post_processing.callback_factory import CallbackFactory
from .common.post_processing.mini_batch_context import MiniBatchContext
from .common.post_processing.result_utils import (
    get_return_value,
    save_mini_batch_results,
)
from .common.request_modification.input_transformer import InputTransformer
from .common.request_modification.modifiers.prompt_redaction_modifier import (
    PromptRedactionModifier,
)
from .common.request_modification.modifiers.request_modifier import RequestModifier
from .common.request_modification.modifiers.vesta_chat_completion_encoded_image_scrubber import (
    VestaChatCompletionEncodedImageScrubber,
)
from .common.request_modification.modifiers.vesta_chat_completion_image_modifier import (
    VestaChatCompletionImageModifier,
)
from .common.request_modification.modifiers.vesta_encoded_image_scrubber import (
    VestaEncodedImageScrubber,
)
from .common.request_modification.modifiers.vesta_image_encoder import ImageEncoder
from .common.request_modification.modifiers.vesta_image_modifier import (
    VestaImageModifier,
)
from .common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from .common.telemetry import logging_utils as lu
from .common.telemetry.logging_utils import (
    get_events_client,
    get_logger,
    set_batch_pool,
    set_mini_batch_id,
    set_quota_audience,
    setup_logger,
)
from .common.telemetry.trace_configs import (
    ConnectionCreateEndTrace,
    ConnectionCreateStartTrace,
    ConnectionReuseconnTrace,
    ExceptionTrace,
    RequestEndTrace,
    RequestRedirectTrace,
    ResponseChunkReceivedTrace,
)
from .header_handlers.meds.meds_header_handler import MedsHeaderHandler
from .header_handlers.open_ai.chat_completion_header_handler import ChatCompletionHeaderHandler
from .header_handlers.open_ai.completion_header_handler import CompletionHeaderHandler
from .header_handlers.open_ai.open_ai_header_handler import OpenAIHeaderHandler
from .header_handlers.open_ai.sahara_header_handler import SaharaHeaderHandler
from .header_handlers.open_ai.vesta_header_handler import VestaHeaderHandler
from .header_handlers.mir_endpoint_v2_header_handler import MIREndpointV2HeaderHandler
from .header_handlers.open_ai.vesta_chat_completion_header_handler import (
    VestaChatCompletionHeaderHandler,
)
from .header_handlers.rate_limiter.rate_limiter_header_handler import (
    RateLimiterHeaderHandler,
)
from .utils.common import convert_to_list, str2bool
from .utils.json_encoder_extensions import setup_encoder

par: parallel.Parallel = None
configuration: Configuration = None


def init():
    """Init function of the component."""
    global par
    global configuration

    print("Entered init")

    configuration = ConfigurationParser().parse_configuration()

    setup_logger("DEBUG" if configuration.debug_mode else "INFO", configuration.app_insights_connection_string)
    configuration.log()

    setup_encoder(configuration.ensure_ascii)

    token_provider = TokenProvider(token_file_path=configuration.token_file_path)
    routing_client = setup_routing_client(token_provider=token_provider)
    scoring_client = setup_scoring_client(configuration=configuration,
                                          token_provider=token_provider,
                                          routing_client=routing_client)
    trace_configs = setup_trace_configs()
    input_to_request_transformer = setup_input_to_request_transformer()
    input_to_log_transformer = setup_input_to_log_transformer()
    input_to_output_transformer = setup_input_to_output_transformer()

    loop = setup_loop()
    finished_callback = None
    if configuration.async_mode:
        callback_factory = CallbackFactory(
            configuration=configuration,
            input_to_output_transformer=input_to_output_transformer)
        finished_callback = callback_factory.generate_callback()

    conductor = parallel.Conductor(
        configuration=configuration,
        loop=loop,
        routing_client=routing_client,
        scoring_client=scoring_client,
        trace_configs=trace_configs,
        finished_callback=finished_callback,
    )

    par = parallel.Parallel(
        configuration=configuration,
        loop=loop,
        conductor=conductor,
        input_to_request_transformer=input_to_request_transformer,
        input_to_log_transformer=input_to_log_transformer,
        input_to_output_transformer=input_to_output_transformer,
    )

    get_events_client().emit_batch_driver_init(job_params=vars(configuration))


def run(input_data: pd.DataFrame, mini_batch_context):
    """Run function of the component. Used in sync mode only."""
    global par
    global configuration

    lu.get_logger().info(f"Scoring a new subset of data, length {input_data.shape[0]}...")

    set_mini_batch_id(mini_batch_context.minibatch_index)

    ret = []

    data_list = convert_to_list(input_data,
                                additional_properties=configuration.additional_properties,
                                batch_size_per_request=configuration.batch_size_per_request)

    get_events_client().emit_mini_batch_started(input_row_count=len(data_list))

    try:
        ret = par.run(data_list)

        if configuration.save_mini_batch_results == "enabled":
            lu.get_logger().info("save_mini_batch_results is enabled")
            save_mini_batch_results(ret, configuration.mini_batch_results_out_directory, mini_batch_context)
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
        lu.get_logger().info(f"Completed subset of data, length {input_data.shape[0]}.")
        set_mini_batch_id(None)

    return get_return_value(ret, configuration.output_behavior)


def enqueue(input_data: pd.DataFrame, mini_batch_context):
    """Enqueue function of the component. Used in async mode only."""
    global par
    global configuration

    mini_batch_id = mini_batch_context.minibatch_index
    lu.get_logger().info("Enqueueing new mini-batch {}...".format(mini_batch_id))

    set_mini_batch_id(mini_batch_id)

    data_list = convert_to_list(input_data,
                                additional_properties=configuration.additional_properties,
                                batch_size_per_request=configuration.batch_size_per_request)

    get_events_client().emit_mini_batch_started(input_row_count=len(data_list))
    lu.get_logger().info("Number of payloads in this mini-batch: {}".format(len(data_list)))

    mini_batch_context = MiniBatchContext(mini_batch_context, len(data_list))

    try:
        par.enqueue(data_list, mini_batch_context)
    except Exception as e:
        get_events_client().emit_mini_batch_completed(
            input_row_count=len(data_list),
            output_row_count=0,
            exception=type(e).__name__,
            stacktrace=traceback.format_exception(etype=type(e), value=e, tb=e.__traceback__)
        )
        raise e
    finally:
        lu.get_logger().info("Completed enqueueing mini-batch {}.".format(mini_batch_id))
        set_mini_batch_id(None)


def check_all_tasks_processed() -> bool:
    """Check if all enqueued tasks are processed. Used in async mode only."""
    global par
    global configuration

    lu.get_logger().debug("check_all_tasks_processed")
    return par.check_all_tasks_processed()


def get_finished_batch_result() -> "dict[str, dict[str, any]]":
    """Get result of finished mini batches. Used in async mode only."""
    global par
    global configuration

    lu.get_logger().debug("get_finished_batch_result")
    return par.get_finished_batch_result()


def get_processing_batch_number() -> int:
    """Get number of mini batches are being processed. Used in async mode only."""
    global par
    global configuration

    lu.get_logger().debug("get_processing_batch_number")
    return par.get_processing_batch_number()


def shutdown():
    """Shutdown function of the component."""
    global par
    global configuration

    lu.get_logger().info("Starting shutdown")
    get_events_client().emit_batch_driver_shutdown(job_params=vars(configuration))

    par.shutdown()


def setup_trace_configs():
    """Set up trace log configurations."""
    is_enabled = os.environ.get(constants.BATCH_SCORE_TRACE_LOGGING, None)
    trace_configs = None

    if is_enabled and is_enabled.lower() == "true":
        lu.get_logger().info("Trace logging enabled, populating trace_configs.")
        trace_configs = [ExceptionTrace(),
                         ResponseChunkReceivedTrace(),
                         RequestEndTrace(),
                         RequestRedirectTrace(),
                         ConnectionCreateStartTrace(),
                         ConnectionCreateEndTrace(),
                         ConnectionReuseconnTrace()]
    else:
        lu.get_logger().info("Trace logging disabled")

    return trace_configs


def setup_loop() -> asyncio.AbstractEventLoop:
    """Set up event loop."""
    if sys.platform == 'win32':
        # For windows environment
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    return asyncio.new_event_loop()


def setup_input_to_request_transformer() -> InputTransformer:
    """Set up the tranformer used to modify each row of the input data before it is sent to MIR for scoring."""
    modifiers: "list[RequestModifier]" = []
    if configuration.is_vesta():
        modifiers.append(VestaImageModifier(
            image_encoder=ImageEncoder(image_input_folder_str=configuration.image_input_folder)))
    elif configuration.is_vesta_chat_completion():
        modifiers.append(VestaChatCompletionImageModifier(
            image_encoder=ImageEncoder(image_input_folder_str=configuration.image_input_folder)))

    return InputTransformer(modifiers=modifiers)


def setup_input_to_log_transformer() -> InputTransformer:
    """Set up the tranformer used to modify each row of the input data before it is logged."""
    modifiers: "list[RequestModifier]" = []
    if configuration.is_vesta():
        modifiers.append(VestaEncodedImageScrubber())
    elif configuration.is_vesta_chat_completion():
        modifiers.append(VestaChatCompletionEncodedImageScrubber())

    if not _should_emit_prompts_to_job_log():
        modifiers.append(PromptRedactionModifier())

    return InputTransformer(modifiers=modifiers)


def setup_input_to_output_transformer() -> InputTransformer:
    """Set up the tranformer used to modify each row of the input data before it written to output."""
    modifiers: "list[RequestModifier]" = []
    if configuration.is_vesta():
        modifiers.append(VestaEncodedImageScrubber())
    elif configuration.is_vesta_chat_completion():
        modifiers.append(VestaChatCompletionEncodedImageScrubber())

    return InputTransformer(modifiers=modifiers)


def setup_scoring_client(
    configuration: Configuration,
    token_provider: TokenProvider,
    routing_client: RoutingClient,
) -> "ScoringClient | AoaiScoringClient":
    """Set up scoring client."""
    if (configuration.is_aoai_endpoint() or configuration.is_serverless_endpoint()):
        get_logger().info("Using LLM scoring client.")
        return setup_aoai_scoring_client(configuration=configuration)
    else:
        get_logger().info("Using MIR scoring client.")
        return setup_mir_scoring_client(
            token_provider=token_provider,
            routing_client=routing_client,
            connection_name=configuration.connection_name
        )


def setup_aoai_scoring_client(
    configuration: Configuration
) -> AoaiScoringClient:
    """Set up AOAI scoring client."""
    tally_handler = TallyFailedRequestHandler(
        enabled=configuration.tally_failed_requests,
        tally_exclusions=configuration.tally_exclusions
    )

    scoring_client = AoaiScoringClient.create(configuration, tally_handler)
    scoring_client.validate()

    return scoring_client


def setup_mir_scoring_client(
    token_provider: TokenProvider,
    routing_client: RoutingClient,
    connection_name: str
) -> ScoringClient:
    """Set up MIR scoring client."""
    quota_client: QuotaClient = None
    if configuration.batch_pool and configuration.quota_audience and configuration.service_namespace:
        set_batch_pool(configuration.batch_pool)
        set_quota_audience(configuration.quota_audience)

        rate_limiter_header_handler = RateLimiterHeaderHandler(
            token_provider=token_provider,
            user_agent_segment=configuration.user_agent_segment,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience)

        quota_client = QuotaClient(
            header_handler=rate_limiter_header_handler,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience,
            quota_estimator=configuration.quota_estimator,
            service_namespace=configuration.service_namespace
        )

    tally_handler = TallyFailedRequestHandler(
        enabled=configuration.tally_failed_requests,
        tally_exclusions=configuration.tally_exclusions
    )

    header_handler = None

    """
    Get the auth token from workspace connection and add that to the header.
    Additionally, add the 'Content-Type' header.
    The presence of workspace connection also signifies that this can be any MIR endpoint,
    so this will not add OAI specific headers.
    """
    if connection_name is not None:
        header_handler = MIREndpointV2HeaderHandler(connection_name)
    else:
        header_handler = setup_header_handler(routing_client=routing_client, token_provider=token_provider)

    scoring_client = ScoringClient(
        header_handler=header_handler,
        quota_client=quota_client,
        routing_client=routing_client,
        online_endpoint_url=configuration.online_endpoint_url,
        tally_handler=tally_handler,
    )

    return scoring_client


def setup_routing_client(token_provider: TokenProvider) -> RoutingClient:
    """Set up routing client."""
    routing_client: RoutingClient = None
    if configuration.batch_pool and configuration.service_namespace:
        meds_header_handler = MedsHeaderHandler(
            token_provider=token_provider,
            user_agent_segment=configuration.user_agent_segment,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience)

        routing_client = RoutingClient(
            header_handler=meds_header_handler,
            service_namespace=configuration.service_namespace,
            target_batch_pool=configuration.batch_pool,
            request_path=configuration.request_path,
        )

    return routing_client


def setup_header_handler(routing_client: RoutingClient, token_provider: TokenProvider) -> OpenAIHeaderHandler:
    """Set up header handler."""
    if configuration.is_sahara(routing_client=routing_client):
        return SaharaHeaderHandler(token_provider=token_provider,
                                   user_agent_segment=configuration.user_agent_segment,
                                   batch_pool=configuration.batch_pool,
                                   quota_audience=configuration.quota_audience,
                                   additional_headers=configuration.additional_headers)
    if configuration.is_vesta():
        return VestaHeaderHandler(token_provider=token_provider,
                                  user_agent_segment=configuration.user_agent_segment,
                                  batch_pool=configuration.batch_pool,
                                  quota_audience=configuration.quota_audience,
                                  additional_headers=configuration.additional_headers)
    if configuration.is_vesta_chat_completion():
        return VestaChatCompletionHeaderHandler(token_provider=token_provider,
                                                user_agent_segment=configuration.user_agent_segment,
                                                batch_pool=configuration.batch_pool,
                                                quota_audience=configuration.quota_audience,
                                                additional_headers=configuration.additional_headers)
    # TODO: Embeddings should probably have its own handler
    if configuration.is_completion() or configuration.is_embeddings():
        return CompletionHeaderHandler(token_provider=token_provider,
                                       user_agent_segment=configuration.user_agent_segment,
                                       batch_pool=configuration.batch_pool,
                                       quota_audience=configuration.quota_audience,
                                       additional_headers=configuration.additional_headers)
    if configuration.is_chat_completion():
        return ChatCompletionHeaderHandler(token_provider=token_provider,
                                           user_agent_segment=configuration.user_agent_segment,
                                           batch_pool=configuration.batch_pool,
                                           quota_audience=configuration.quota_audience,
                                           additional_headers=configuration.additional_headers)

    get_logger().info("No OpenAI model matched, defaulting to base OpenAI header handler.")
    return OpenAIHeaderHandler(token_provider=token_provider,
                               user_agent_segment=configuration.user_agent_segment,
                               batch_pool=configuration.batch_pool,
                               quota_audience=configuration.quota_audience,
                               additional_headers=configuration.additional_headers)


def _should_emit_prompts_to_job_log() -> bool:
    emit_prompts_to_job_log_env_var = os.environ.get(constants.BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG)
    if emit_prompts_to_job_log_env_var is None:
        # Disable emitting prompts to job log by default for LLM compone
        # keep it enabled by default for all other components.
        if configuration.scoring_url is not None:
            emit_prompts_to_job_log_env_var = "False"
        else:
            emit_prompts_to_job_log_env_var = "True"

    should_emit_prompts_to_job_log = str2bool(emit_prompts_to_job_log_env_var)
    lu.get_logger().info("Emitting prompts to job log is {}.".format(
        "enabled" if should_emit_prompts_to_job_log else "disabled"))
