# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""The class for scoring client."""

from typing import Any
import traceback
from datetime import datetime, timezone
import aiohttp
import time
import asyncio
import uuid

from azureml._common._error_definition.azureml_error import AzureMLError
from ..utils.exceptions import BenchmarkValidationException
from ..utils.error_definitions import BenchmarkValidationError

from .scoring_result import RetriableException, ScoringResult, ScoringResultStatus
from .scoring_request import ScoringRequest
from .tally_failed_request_handler import TallyFailedRequestHandler
from ..header_handlers.header_handler import HeaderHandler
from .common.common import get_base_url
from . import logging_utils as lu
from .logging_utils import get_events_client
from . import scoring_utils


class ScoringClient:
    """Class for scoring client."""

    DV_COMPLETION_API_PATH = "v1/engines/davinci/completions"
    DV_EMBEDDINGS_API_PATH = "v1/engines/davinci/embeddings"
    DV_CHAT_COMPLETIONS_API_PATH = "v1/engines/davinci/chat/completions"
    VESTA_RAINBOW_API_PATH = "v1/rainbow"

    def __init__(
            self,
            header_handler: HeaderHandler,
            quota_client: Any = None,
            routing_client: Any = None,
            online_endpoint_url: str = None,
            tally_handler: TallyFailedRequestHandler = None):
        """Init method."""
        self.__header_handler = header_handler
        self.__routing_client = routing_client
        self.__quota_client = quota_client
        self.__online_endpoint_url: str = online_endpoint_url
        self.__tally_handler = tally_handler

        # Check for invalid parameter combination
        if self.__routing_client is not None and \
            (self.__online_endpoint_url is not None or
             "azureml-model-deployment" in self.__header_handler.get_headers()):
            lu.get_logger().error(
                "Invalid parameter combination. batch_pool AND "
                "(online_endpoint_url or azureml-model-deployment header) are provided.")
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(
                    BenchmarkValidationError, error_details="Invalid parameter combination")
            )

    async def score_until_completion(
            self,
            session: aiohttp.ClientSession,
            scoring_request: ScoringRequest,
            worker_id: str = "1"
    ) -> ScoringResult:
        """Score until completion method."""
        cur_retries = 0
        base_wait = 1
        scoring_result: ScoringResult = None

        start = time.time()

        while scoring_result is None:
            try:
                scoring_result = await self.score_once(
                    session=session, scoring_request=scoring_request, worker_id=worker_id)
            except RetriableException as e:
                cur_retries = cur_retries + 1
                retry_wait = e.retry_after or base_wait * min(cur_retries, 5)
                lu.get_logger().debug("{}: retrying in {} seconds...".format(worker_id, retry_wait))
                await asyncio.sleep(retry_wait)

        end = time.time()

        # Since we're scoring until termination, update start and end to overall start and end
        scoring_result.start = start
        scoring_result.end = end
        scoring_result.num_retries = cur_retries

        return scoring_result

    async def score_once(
            self,
            session: aiohttp.ClientSession,
            scoring_request: ScoringRequest,
            worker_id: str = "1"
    ) -> ScoringResult:
        """Score once method."""
        if self.__routing_client and self.__quota_client:
            quota_scope = await self.__routing_client.get_quota_scope(session)

            async with self.__quota_client.reserve_capacity(session, quota_scope, scoring_request) as lease:
                result = await self._score_once(session, scoring_request, worker_id)
                lease.report_result(result)
                return result
        else:
            result = await self._score_once(
                session=session, scoring_request=scoring_request, worker_id=worker_id)
            return result

    async def _score_once(
            self,
            session: aiohttp.ClientSession,
            scoring_request: ScoringRequest,
            worker_id: str = "1"
    ) -> ScoringResult:
        response = None
        response_payload = None
        response_status = None
        response_headers = None
        model_response_code = None
        model_response_reason = None
        client_exception = None

        headers = self.__header_handler.get_headers(payload=scoring_request.cleaned_payload)

        target_endpoint_url = self.__online_endpoint_url

        if self.__routing_client is not None:
            exclude_endpoint = None
            # If the most recent response_status for this request was insufficient,
            # exclude the corresponding endpoint on this next attempt
            # 404 case is to exclude an endpoint that encountered ZeroTrafficGroupError
            if len(scoring_request.request_history) > 0 and (
                scoring_request.request_history[-1][1] == 429
                or scoring_request.request_history[-1][2] == "429"
                or (scoring_request.request_history[-1][1] == 404 and
                    scoring_request.request_history[-1][3] is True)):
                exclude_endpoint = scoring_request.request_history[-1][0]
                lu.get_logger().debug(
                    "{}: Excluding endpoint '{}' from consideration "
                    "for the next attempt of this scoring_request".format(worker_id, exclude_endpoint))

            target_endpoint_url = await self.__routing_client.get_target_endpoint(
                session=session, exclude_endpoint=exclude_endpoint)

        endpoint_base_url = get_base_url(target_endpoint_url)

        # lu.get_logger().info("{}: Score start: {}, url={} internal_id={} x-ms-client-request-id=[{}]".format(
        #     worker_id,
        #     datetime.fromtimestamp(start, timezone.utc),
        #     target_endpoint_url,
        #     scoring_request.internal_id,
        #     headers["x-ms-client-request-id"]))

        lu.get_logger().info("{}: url={} internal_id={} x-ms-client-request-id=[{}]".format(
            worker_id,
            target_endpoint_url,
            scoring_request.internal_id,
            headers["x-ms-client-request-id"]))

        payload_correlation_id = str(uuid.uuid4())
        lu.get_logger().debug(
            "{}: Score request payload: <Redacted from application insights logging;"
            " see job logs directly. payload_correlation_id: {}>".format(
                worker_id, payload_correlation_id))
        lu.get_logger().debug("{}: Score request payload [payload_correlation_id: {}]: {}".format(
            worker_id, payload_correlation_id, scoring_request.loggable_payload))

        start = time.time()

        try:
            if self.__routing_client is not None:
                self.__routing_client.increment(endpoint_base_url, scoring_request)

            async with session.post(
                url=target_endpoint_url,
                headers=headers, data=scoring_request.cleaned_payload,
                trace_request_ctx={"worker_id": worker_id}
            ) as response:
                response_status = response.status
                response_headers = response.headers
                model_response_code = response_headers.get("ms-azureml-model-error-statuscode")
                model_response_reason = response_headers.get("ms-azureml-model-error-reason")

                if response_status == 200:
                    response_payload = await response.json()
                else:
                    response_payload = await response.text()
                    lu.get_logger().error(
                        "{}: Score failed -- internal_id: {} status: {} reason: {} "
                        "-- response headers: {} | response payload: {}".format(
                            worker_id,
                            scoring_request.internal_id,
                            "NONE" if response is None else response.status,
                            "NONE" if response is None else response.reason,
                            "NONE" if response is None else response.headers,
                            response_payload))
        except asyncio.TimeoutError as e:
            client_exception = e
            response_status = -408  # Manually attribute -408 as a tell to retry on asyncio exception

            lu.get_logger().error(
                "{}: Score failed: asyncio.TimeoutError -- internal_id: {} x-ms-client-request-id: {}".format(
                    worker_id,
                    scoring_request.internal_id,
                    headers["x-ms-client-request-id"]))
        except aiohttp.ServerConnectionError as e:
            client_exception = e
            response_status = -408  # Manually attribute -408 as a tell to retry on this exception

            lu.get_logger().error(
                "{}: Score failed: aiohttp.ServerConnectionError -- internal_id: {}"
                " x-ms-client-request-id: {}. Exception: {}".format(
                    worker_id,
                    scoring_request.internal_id,
                    headers["x-ms-client-request-id"],
                    e))
        except aiohttp.ClientConnectorError as e:
            client_exception = e
            response_status = -408  # Manually attribute -408 as a tell to retry on this exception

            lu.get_logger().error(
                "{}: Score failed: aiohttp.ClientConnectorError -- x-ms-client-request-id: {}."
                " Exception: {}".format(
                    worker_id,
                    headers["x-ms-client-request-id"],
                    e))
        except Exception as e:
            client_exception = e
            # Manually attribute -500 as a tell to not retry unhandled exceptions
            response_status = -500

            lu.get_logger().error(
                "{}: Score failed: unhandled exception -- internal_id: {} x-ms-client-request-id: {}."
                " Exception: {}".format(
                    worker_id,
                    scoring_request.internal_id,
                    headers["x-ms-client-request-id"],
                    traceback.format_exc()))
        finally:
            if self.__routing_client is not None:
                self.__routing_client.decrement(endpoint_base_url, scoring_request)

        end = time.time()

        is_retriable = scoring_utils.is_retriable(response_status=response_status,
                                                  response_payload=response_payload,
                                                  model_response_code=model_response_code,
                                                  model_response_reason=model_response_reason)

        # Record endpoint url and response_status
        scoring_request.request_history.append(
            (endpoint_base_url, response_status, model_response_code, is_retriable))

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

        result: ScoringResult = None

        if response_status == 200:
            lu.get_logger().info(
                "{}: Score succeeded at {} -- internal_id: {} x-ms-client-request-id: {} "
                "total_wait_time: {}".format(
                    worker_id,
                    datetime.fromtimestamp(end, timezone.utc),
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
                num_retries=0
            )
            get_events_client().emit_tokens_generated(
                result.completion_tokens, result.prompt_tokens, target_endpoint_url)
        elif is_retriable:
            raise RetriableException(
                status_code=response_status, response_payload=response_payload,
                model_response_code=model_response_code, model_response_reason=model_response_reason)
        else:  # Score failed
            result = ScoringResult(
                status=ScoringResultStatus.FAILURE,
                start=start,
                end=end,
                request_obj=scoring_request.original_payload_obj,
                request_metadata=scoring_request.request_metadata,
                response_body=response_payload,
                response_headers=response_headers,
                num_retries=0
            )

        if result.status == ScoringResultStatus.FAILURE and \
                self.__tally_handler.should_tally(
                    response_status=response_status, model_response_status=model_response_code):
            result.omit = True

        return result
