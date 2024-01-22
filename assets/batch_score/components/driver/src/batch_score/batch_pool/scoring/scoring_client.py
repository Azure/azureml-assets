import asyncio
import time
import traceback
import uuid

import aiohttp

from ...common.scoring import scoring_utils
from ...common.scoring.scoring_attempt import ScoringAttempt
from ...common.scoring.scoring_request import ScoringRequest
from ...common.scoring.scoring_result import (
    RetriableException,
    ScoringResult,
    ScoringResultStatus,
)
from ...common.scoring.scoring_utils import RetriableType
from ...common.scoring.tally_failed_request_handler import TallyFailedRequestHandler
from ...common.telemetry import logging_utils as lu
from ...common.telemetry.events import event_utils
from ...common.telemetry.events.batch_score_request_completed_event import BatchScoreRequestCompletedEvent
from ...common.telemetry.logging_utils import get_events_client
from ...header_handlers.open_ai.open_ai_header_handler import OpenAIHeaderHandler
from ...utils.common import get_base_url
from ..quota import QuotaClient
from ..routing.routing_client import RoutingClient


class ScoringClient:
    MINIMUM_SCORING_TIMEOUT = 10

    def __init__(
            self,
            header_handler: OpenAIHeaderHandler,
            quota_client: QuotaClient,
            routing_client: RoutingClient = None,
            scoring_url: str = None,
            tally_handler: TallyFailedRequestHandler = None):
        self.__header_handler = header_handler
        self.__routing_client = routing_client
        self.__quota_client = quota_client
        self.__scoring_url: str = scoring_url
        self.__tally_handler = tally_handler

        self._validate_init_params()

    async def score_once(
            self,
            session: aiohttp.ClientSession,
            scoring_request: ScoringRequest,
            timeout: aiohttp.ClientTimeout = None,
            worker_id: str = "1"
    ) -> ScoringResult:
        if self.__routing_client and self.__quota_client:
            quota_scope = await self.__routing_client.get_quota_scope(session)

            async with self.__quota_client.reserve_capacity(session, quota_scope, scoring_request) as lease:
                result = await self._score_once(session, scoring_request, timeout, worker_id)
                lease.report_result(result)
                return result
        else:
            result = await self._score_once(
                session=session,
                timeout=timeout,
                scoring_request=scoring_request,
                worker_id=worker_id)
            return result

    async def _score_once(
            self,
            session: aiohttp.ClientSession,
            scoring_request: ScoringRequest,
            timeout: aiohttp.ClientTimeout = None,
            worker_id: str = "1"
    ) -> ScoringResult:
        response = None
        response_payload = None
        response_status = None
        response_headers = None
        model_response_code = None
        model_response_reason = None
        client_exception = None

        # Timeout can be None. See `ScoringClient.get_next_retry_timeout` for more info on why.
        if timeout is None:
            timeout = session.timeout

        lu.get_logger().debug("Worker_id: {}, internal_id: {}, Timeout: {}s".format(worker_id, scoring_request.internal_id, timeout.total))

        start = time.time()

        target_endpoint_url = await self._get_target_endpoint_url(session, scoring_request, worker_id)
        scoring_request.scoring_url = target_endpoint_url
        
        endpoint_base_url = get_base_url(target_endpoint_url)
        headers = self.__header_handler.get_headers()
        self._log_score_start(scoring_request, worker_id, target_endpoint_url, headers)

        try:
            if self.__routing_client is not None:
                self.__routing_client.increment(endpoint_base_url, scoring_request)

            async with session.post(
                    url=target_endpoint_url,
                    headers=headers,
                    data=scoring_request.cleaned_payload,
                    trace_request_ctx={"worker_id": worker_id},
                    timeout=timeout) as response:
                response_status = response.status
                response_headers = response.headers
                model_response_code = response_headers.get("ms-azureml-model-error-statuscode")
                model_response_reason = response_headers.get("ms-azureml-model-error-reason")

                if response_status == 200:
                    response_payload = await response.json()
                else:
                    response_payload = await response.text()
                    lu.get_logger().error("Worker {}: Score failed -- internal_id: {} status: {} reason: {} -- response headers: {} | response payload: {}".format(
                        worker_id,
                        scoring_request.internal_id,
                        "NONE" if response is None else response.status,
                        "NONE" if response is None else response.reason,
                        "NONE" if response is None else response.headers,
                        response_payload))

        except (asyncio.TimeoutError, aiohttp.ServerConnectionError, aiohttp.ClientConnectionError, aiohttp.ClientPayloadError) as e:
            client_exception = e
            fully_qualified_exception_name = type(client_exception).__module__ + "." + type(client_exception).__name__
            response_status = -408 # Manually attribute -408 as a tell to retry on this exception

            lu.get_logger().error("Worker {}: Score failed: {} -- internal_id: {} x-ms-client-request-id: {}".format(
                worker_id, 
                fully_qualified_exception_name,
                scoring_request.internal_id,
                headers["x-ms-client-request-id"]))
        except Exception as e:
            client_exception = e
            response_status = -500 # Manually attribute -500 as a tell to not retry unhandled exceptions

            lu.get_logger().error("Worker {}: Score failed: unhandled exception -- internal_id: {} x-ms-client-request-id: {}. Exception: {}".format(
                worker_id,
                scoring_request.internal_id,
                headers["x-ms-client-request-id"],
                traceback.format_exc()))

        finally:
            if self.__routing_client is not None:
                self.__routing_client.decrement(endpoint_base_url, scoring_request)

        end = time.time()

        retriable_type = scoring_utils.get_retriable_type(response_status=response_status,
                                                  response_payload=response_payload,
                                                  model_response_code=model_response_code,
                                                  model_response_reason=model_response_reason)
        
        is_retriable = scoring_utils.is_retriable(retriable_type)

        # Record endpoint url and response_status
        scoring_request.request_history.append(ScoringAttempt(
            endpoint_base_url=endpoint_base_url,
            response_status=response_status,
            model_response_code=model_response_code,
            retriable_type=retriable_type,
        ))

        # TODO: Clean up this log event
        get_events_client().emit_request_completed(
            latency=(end-start) * 1000,
            request_internal_id=scoring_request.internal_id,
            client_request_id=headers["x-ms-client-request-id"],
            endpoint_uri=target_endpoint_url,
            status_code="0" if response_status is None else str(response_status),
            model_response_code="" if model_response_code is None else model_response_code,
            client_exception="" if client_exception is None else type(client_exception).__name__,
            is_retriable=is_retriable
        )

        # Emit request completed event (TO DO: Move this to a method) 
        def get_prompt_tokens(response_body: any):
            if response_body and not isinstance(response_body, list):
                usage: dict[str, int] = response_body["usage"]
                return usage.get("prompt_tokens", None)
        
        def get_completion_tokens(response_body: any):
            if response_body and not isinstance(response_body, list):
                usage: dict[str, int] = response_body["usage"]
                return usage.get("completion_tokens", None)
               
        request_completed_event = BatchScoreRequestCompletedEvent(
            minibatch_id = scoring_request.mini_batch_context.mini_batch_id if scoring_request.mini_batch_context is not None else None,
            input_row_id = scoring_request.internal_id,
            x_ms_client_request_id = headers["x-ms-client-request-id"],
            worker_id = worker_id,
            scoring_url = target_endpoint_url,
            is_successful = response_status == 200,
            is_retriable = is_retriable,
            response_code = response_status,
            model_response_code = model_response_code,
            prompt_tokens = get_prompt_tokens(response_payload),
            completion_tokens = get_completion_tokens(response_payload),
            duration_ms = (end - start) * 1000,
            segmented_request_id = scoring_request.segment_id
        )
        event_utils.emit_event(batch_score_event = request_completed_event)

        result: ScoringResult = None

        if response_status == 200:
            lu.get_logger().info("Worker {}: Score succeeded after {:.3f}s -- internal_id: {} x-ms-client-request-id: {} total_wait_time: {}".format(
                worker_id,
                end - start,
                scoring_request.internal_id,
                headers["x-ms-client-request-id"],
                scoring_request.total_wait_time))

            result = ScoringResult(
                status=ScoringResultStatus.SUCCESS,
                start=start,
                end=end,
                request_obj=scoring_request.original_payload_obj,
                request_metadata=scoring_request.request_metadata,
                response_body=response_payload,
                response_headers=response_headers,
                num_retries=0,
                token_counts=scoring_request.estimated_token_counts,
                mini_batch_context=scoring_request.mini_batch_context
            )

            get_events_client().emit_tokens_generated(result.completion_tokens, result.prompt_tokens, target_endpoint_url)
        elif is_retriable:
            raise RetriableException(status_code=response_status, response_payload=response_payload, model_response_code=model_response_code, model_response_reason=model_response_reason)
        else: # Score failed
            result = ScoringResult(
                status=ScoringResultStatus.FAILURE,
                start=start,
                end=end,
                request_obj=scoring_request.original_payload_obj,
                request_metadata=scoring_request.request_metadata,
                response_body=response_payload,
                response_headers=response_headers,
                num_retries=0,
                mini_batch_context=scoring_request.mini_batch_context
            )

        if result.status == ScoringResultStatus.FAILURE and self.__tally_handler.should_tally(response_status=response_status, model_response_status=model_response_code):
            result.omit = True

        return result

    def get_next_retry_timeout(timeout_generator):
        try:
            return next(timeout_generator)
        except StopIteration:
            # We may encounter this if there is no max_retry_time_interval configured to stop attempting to
            # process the queue item when scoring duration continues to grow. In that case, the customer
            # wants to allow the score to retry forever. Setting the timeout to None will default to use the
            # aiohttp.ClientSession's timeout, which is set in `Conductor.__configure_session_timeout`
            return None


    def get_retry_timeout_generator(default_timeout: aiohttp.ClientTimeout):
        """
        A Python generator that yields aiohttp.ClientTimeout objects.
        """
        for iteration in range(2, 10):
            timeout = max(ScoringClient.MINIMUM_SCORING_TIMEOUT, int(2.7 ** iteration))
            if timeout >= default_timeout.total:
                break
            else:
                yield aiohttp.ClientTimeout(timeout)
        yield default_timeout

    async def _get_target_endpoint_url(self, session, scoring_request, worker_id):
        if self.__routing_client is None:
            return self.__scoring_url

        exclude_endpoint = None
        # If the most recent response_status for this request was insufficient, exclude the corresponding endpoint on this next attempt
        if len(scoring_request.request_history) > 0:
            latest_attempt = scoring_request.request_history[-1]
            if latest_attempt.retriable_type == RetriableType.RETRY_ON_DIFFERENT_ENDPOINT:
                exclude_endpoint = latest_attempt.endpoint_base_url
                lu.get_logger().debug("{}: Excluding endpoint '{}' from consideration for the next attempt of this scoring_request".format(worker_id, exclude_endpoint))

        target_endpoint_url = await self.__routing_client.get_target_endpoint(session=session, exclude_endpoint=exclude_endpoint)
        return target_endpoint_url

    def _log_score_start(self, scoring_request, worker_id, target_endpoint_url, headers):
        # Redact bearer token before logging
        redacted_headers = headers.copy()
        redacted_headers["Authorization"] = "Redacted"

        lu.get_logger().info("Worker {}: Score start -- url={} internal_id={} x-ms-client-request-id=[{}] request headers={}".format(
            worker_id,
            target_endpoint_url,
            scoring_request.internal_id,
            headers["x-ms-client-request-id"],
            redacted_headers))

        payload_correlation_id = str(uuid.uuid4())
        lu.get_logger().debug("AppInsRedact: Worker {}: Score request payload [payload_correlation_id: {}]: {}".format(
            worker_id, payload_correlation_id, scoring_request.loggable_payload))

    def _validate_init_params(self):
        # Check for invalid parameter combination
        '''Even though we check for `scoring_url` the error message says `online_endpoint_url` instead.
        This is because the below condition can occur only in batch-pool mode and the component that supports batch-pool mode exposes `online_endpoint_url` and not `scoring_url`.
        '''
        if self.__routing_client is not None and \
            (self.__scoring_url is not None or "azureml-model-deployment" in self.__header_handler.get_headers()):
            lu.get_logger().error("Invalid parameter combination. batch_pool AND (online_endpoint_url or azureml-model-deployment header) are provided.")

            raise Exception("Invalid parameter combination")