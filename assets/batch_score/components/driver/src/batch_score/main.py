import asyncio
import os
import sys
import time
import traceback

import pandas as pd

from .batch_pool.routing.routing_client import RoutingClient
from .common import constants
from .common.auth.token_provider import TokenProvider
from .common.configuration.configuration import Configuration
from .common.configuration.metadata import Metadata
from .common.configuration.configuration_parser_factory import ConfigurationParserFactory
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
from .common.scoring.scoring_client_factory import ScoringClientFactory
from .common.telemetry import logging_utils as lu
from .common.telemetry.event_listeners.geneva_event_listener import setup_geneva_event_handlers
from .common.telemetry.event_listeners.job_log_event_listener import setup_job_log_event_handlers
from .common.telemetry.events import event_utils
from .common.telemetry.events.batch_score_init_started_event import BatchScoreInitStartedEvent
from .common.telemetry.events.batch_score_init_completed_event import BatchScoreInitCompletedEvent
from .common.telemetry.events.batch_score_minibatch_started_event import BatchScoreMinibatchStartedEvent
from .common.telemetry.logging_utils import (
    get_events_client,
    set_mini_batch_id,
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
from .utils.common import convert_to_list, str2bool
from .utils.json_encoder_extensions import setup_encoder

par: parallel.Parallel = None
configuration: Configuration = None

def init():
    global par
    global configuration

    start = time.time()
    parser = ConfigurationParserFactory().get_parser()
    configuration = parser.parse_configuration()
    metadata = Metadata.extract_component_name_and_version()

    event_utils.setup_context_vars(configuration, metadata)
    setup_geneva_event_handlers()
    setup_job_log_event_handlers()
    setup_logger(configuration.stdout_log_level,
                 configuration.app_insights_log_level,
                 configuration.app_insights_connection_string)
    
    # Emit init started event
    init_started_event = BatchScoreInitStartedEvent()
    event_utils.emit_event(batch_score_event = init_started_event)    

    configuration.log()

    setup_encoder(configuration.ensure_ascii)

    token_provider = TokenProvider(token_file_path=configuration.token_file_path)
    routing_client = setup_routing_client(token_provider=token_provider)
    scoring_client = ScoringClientFactory().setup_scoring_client(configuration=configuration,
                                                                 metadata=metadata,
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

    end = time.time()

    # Emit init completed event
    init_completed_event = BatchScoreInitCompletedEvent(init_duration_ms = (end - start) * 1000)
    event_utils.emit_event(batch_score_event = init_completed_event)

    get_events_client().emit_batch_driver_init(job_params=vars(configuration))

def run(input_data: pd.DataFrame, mini_batch_context):
    global par
    global configuration

    lu.get_logger().info(f"Scoring a new data subset, length {input_data.shape[0]}...")
    event_utils.emit_event(
        batch_score_event=BatchScoreMinibatchStartedEvent(
            minibatch_id=mini_batch_context.minibatch_index,
            scoring_url=configuration.scoring_url,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience,
            input_row_count=input_data.shape[0],
        )
    )

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
        event_utils.generate_minibatch_summary(
            minibatch_id=mini_batch_context.minibatch_index,
            output_row_count=len(ret),
        )
        lu.get_logger().info(f"Completed data subset, length {input_data.shape[0]}.")
        set_mini_batch_id(None)

    return get_return_value(ret, configuration.output_behavior)

def enqueue(input_data: pd.DataFrame, mini_batch_context):
    global par
    global configuration

    mini_batch_id = mini_batch_context.minibatch_index
    lu.get_logger().info("Enqueueing new mini-batch {}...".format(mini_batch_id))
    event_utils.emit_event(
        batch_score_event=BatchScoreMinibatchStartedEvent(
            minibatch_id=mini_batch_context.minibatch_index,
            scoring_url=configuration.scoring_url,
            batch_pool=configuration.batch_pool,
            quota_audience=configuration.quota_audience,
            input_row_count=input_data.shape[0],
        )
    )

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
        event_utils.generate_minibatch_summary(
            minibatch_id=mini_batch_id,
            output_row_count=0,
        )
        raise e
    finally:
        lu.get_logger().info("Completed enqueueing mini-batch {}.".format(mini_batch_id))
        set_mini_batch_id(None)

def check_all_tasks_processed() -> bool:
    global par
    global configuration

    lu.get_logger().debug("check_all_tasks_processed")
    return par.check_all_tasks_processed()

def get_finished_batch_result() -> "dict[str, dict[str, any]]":
    global par
    global configuration

    lu.get_logger().debug("get_finished_batch_result")
    return par.get_finished_batch_result()

def get_processing_batch_number() -> int:
    global par
    global configuration

    lu.get_logger().debug("get_processing_batch_number")
    return par.get_processing_batch_number()

def shutdown():
    global par
    global configuration

    lu.get_logger().info("Starting shutdown")
    get_events_client().emit_batch_driver_shutdown(job_params=vars(configuration))

    par.shutdown()

def setup_trace_configs():
    is_enabled = os.environ.get(constants.BATCH_SCORE_TRACE_LOGGING, None)
    trace_configs = None

    if is_enabled and is_enabled.lower() == "true":
        lu.get_logger().info("Trace logging enabled, populating trace_configs.")
        trace_configs = [ExceptionTrace(), ResponseChunkReceivedTrace(), RequestEndTrace(), RequestRedirectTrace(), ConnectionCreateStartTrace(), ConnectionCreateEndTrace(), ConnectionReuseconnTrace()]
    else:
        lu.get_logger().info("Trace logging disabled")

    return trace_configs

def setup_loop() -> asyncio.AbstractEventLoop:
    if sys.platform == 'win32':
        # For windows environment
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    
    return asyncio.new_event_loop()

def setup_input_to_request_transformer() -> InputTransformer:
    """This tranformer is used to modify each row of the input data before it is sent to MIR for scoring."""
    modifiers: "list[RequestModifier]" = []
    if configuration.is_vesta():
        modifiers.append(VestaImageModifier(image_encoder=ImageEncoder(image_input_folder_str=configuration.image_input_folder)))
    elif configuration.is_vesta_chat_completion():
        modifiers.append(VestaChatCompletionImageModifier(image_encoder=ImageEncoder(image_input_folder_str=configuration.image_input_folder)))

    return InputTransformer(modifiers=modifiers)

def setup_input_to_log_transformer() -> InputTransformer:
    """This tranformer is used to modify each row of the input data before it is logged."""
    modifiers: "list[RequestModifier]" = []
    if configuration.is_vesta():
        modifiers.append(VestaEncodedImageScrubber())
    elif configuration.is_vesta_chat_completion():
        modifiers.append(VestaChatCompletionEncodedImageScrubber())
    
    if not _should_emit_prompts_to_job_log():
        modifiers.append(PromptRedactionModifier())

    return InputTransformer(modifiers=modifiers)

def setup_input_to_output_transformer() -> InputTransformer:
    """This tranformer is used to modify each row of the input data before it written to output."""
    modifiers: "list[RequestModifier]" = []
    if configuration.is_vesta():
        modifiers.append(VestaEncodedImageScrubber())
    elif configuration.is_vesta_chat_completion():
        modifiers.append(VestaChatCompletionEncodedImageScrubber())
    
    return InputTransformer(modifiers=modifiers)

def setup_routing_client(token_provider: TokenProvider) -> RoutingClient:
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

def _should_emit_prompts_to_job_log() -> bool:
    emit_prompts_to_job_log_env_var = os.environ.get(constants.BATCH_SCORE_EMIT_PROMPTS_TO_JOB_LOG)
    if emit_prompts_to_job_log_env_var is None:
        # Disable emitting prompts to job log by default for LLM component, 
        # keep it enabled by default for all other components.
        if configuration.scoring_url is not None:
            emit_prompts_to_job_log_env_var = "False"
        else:
            emit_prompts_to_job_log_env_var = "True"
    
    should_emit_prompts_to_job_log = str2bool(emit_prompts_to_job_log_env_var)
    lu.get_logger().info("Emitting prompts to job log is {}.".format("enabled" if should_emit_prompts_to_job_log else "disabled"))
