# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Worker."""

import asyncio
import os
import time
import traceback
from collections import deque

import aiohttp

from ...batch_pool.quota import QuotaUnavailableException
from ...batch_pool.scoring.scoring_client import ScoringClient
from ...utils.common import str2bool
from .. import constants
from ..configuration.client_settings import ClientSettingsKey, ClientSettingsProvider
from ..configuration.configuration import Configuration
from ..scoring.scoring_request import ScoringRequest
from ..scoring.scoring_result import (
    PermanentException,
    RetriableException,
    ScoringResult,
    ScoringResultStatus,
)
from ..scoring.scoring_utils import is_zero_traffic_group_error
from ..scoring.segmented_score_context import SegmentedScoreContext
from ..telemetry import logging_utils as lu
from ..telemetry.logging_utils import set_mini_batch_id
from .request_metrics import RequestMetrics


class QueueItem:
    """Queue Item."""

    def __init__(self,
                 scoring_request: ScoringRequest,
                 segmented_score_context: SegmentedScoreContext = None,
                 timeout_generator=None):
        """Init function."""
        self.scoring_request = scoring_request
        self.segmented_score_context = segmented_score_context
        self.timeout_generator = timeout_generator


class Worker:
    """Worker."""

    NO_DEPLOYMENTS_BACK_OFF = 120

    def __init__(
        self,
        configuration: Configuration,
        scoring_client: ScoringClient,
        client_session: aiohttp.ClientSession,
        client_settings_provider: ClientSettingsProvider,
        scoring_request_queue: "deque[QueueItem]",
        request_metrics: RequestMetrics,
        id: str,
        empty_wait_interval: int = 1,
        scoring_result_queue: "deque[ScoringRequest]" = None,
    ):
        """Init function."""
        self._configuration = configuration
        self.__scoring_client: ScoringClient = scoring_client
        self.__client_session = client_session
        self.__client_settings_provider: ClientSettingsProvider = client_settings_provider
        self.__scoring_request_queue = scoring_request_queue
        self.__request_metrics = request_metrics
        self.id = id
        self.__empty_wait_interval = empty_wait_interval
        self.__scoring_result_queue = scoring_result_queue

        self.__is_running = False
        self.__enable_delay_after_success = str2bool(
            os.environ.get("BATCH_SCORE_DELAY_AFTER_SUCCESSFUL_REQUEST", "True"))

        lu.get_logger().debug("Worker {}: Created".format(self.id))

    def is_running(self) -> bool:
        """Check whether the worker is running."""
        return self.__is_running

    def stop(self):
        """Stop the worker."""
        lu.get_logger().debug("Worker {}: Stopped".format(self.id))
        self.__is_running = False

    async def start(self):
        """Start the worker."""
        self.__is_running = True
        lu.set_worker_id(self.id)
        lu.get_logger().debug("Worker {}: Started".format(self.id))

        while (self._configuration.async_mode or len(self.__scoring_request_queue) > 0) and self.__is_running:
            if len(self.__scoring_request_queue) == 0:
                await asyncio.sleep(self.__empty_wait_interval)
                continue

            queue_item = self.__scoring_request_queue.popleft()

            if self._configuration.async_mode:
                mini_batch_id = queue_item.scoring_request.mini_batch_context.mini_batch_id
                set_mini_batch_id(mini_batch_id)
                lu.get_logger().debug("Worker {}: Picked an queue item from mini-batch {}"
                                      .format(self.id, mini_batch_id))

            start, end = 0, 0

            # If the request's scoring duration exceeds the limit,
            # log as failure and don't add it back to the queue
            if self._request_exceeds_max_retry_time_interval(queue_item.scoring_request):
                result = ScoringResult.Failed(queue_item.scoring_request)
                self.__add_result(queue_item.scoring_request, result)
                self._log_score_failed_with_max_retry_time_interval_exceeded(queue_item.scoring_request)
                continue

            try:
                start = time.time()
                await self.__process_queue_item(queue_item)
                end = time.time()

                if self.__enable_delay_after_success:
                    # On successful request, add a 1 second delay before processing the next one
                    # This is to lower the chance of other requests getting starved because a worker that releases
                    # quota always immediately re-reserve it with no delay.
                    await asyncio.sleep(1)

            except PermanentException as e:
                result = ScoringResult.Failed(queue_item.scoring_request)
                self.__add_result(queue_item.scoring_request, result)
                self._log_score_failed_with_permanent_exception(queue_item.scoring_request, e)

            except RetriableException as e:
                end = time.time()

                # Back-off 1 second before adding ScoringRequest back to the queue
                # TODO: consider adjusting back-off.
                #   - None-429s: increase backoff for consecutive retries
                #   - 429s: vary backoff based on request cost to prevent large request starvation
                back_off = e.retry_after or 1
                queue_item.scoring_request.retry_count += 1
                wait_time = 0

                # To activate this, set BATCH_SCORE_POLL_DURING_NO_DEPLOYMENTS to True.
                # And to override the default back-off time,
                # set BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF to a value in seconds.
                if is_zero_traffic_group_error(e.status_code, e.response_payload) \
                        and str2bool(os.environ.get(constants.BATCH_SCORE_POLL_DURING_NO_DEPLOYMENTS, "False")):
                    back_off = int(
                        os.environ.get(constants.BATCH_SCORE_NO_DEPLOYMENTS_BACK_OFF)
                        or self.__client_settings_provider.get_client_setting(
                            ClientSettingsKey.NO_DEPLOYMENTS_BACK_OFF)
                        or self.NO_DEPLOYMENTS_BACK_OFF)

                    wait_time = back_off
                    queue_item.scoring_request.total_wait_time += wait_time
                    # Reset timeout generator since this error is not due to timeout or model congestion.
                    queue_item.timeout_generator = ScoringClient.get_retry_timeout_generator(
                        self.__client_session.timeout)

                elif e.status_code == 429 or e.model_response_code == "429":
                    wait_time = back_off

                    is_quota_429 = isinstance(e, QuotaUnavailableException)

                    value = self.__client_settings_provider.get_client_setting(
                        ClientSettingsKey.COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME) or "False"
                    count_only_quota_429s_toward_total_request_time: bool = (value.lower() == "true")

                    if count_only_quota_429s_toward_total_request_time and not is_quota_429:
                        # Non-quota 429 responses don't contribute to the total wait time.
                        msg = f"Worker {self.id}: Encountered non-quota 429 response. Not adding to total wait time."
                        lu.get_logger().debug(msg)
                        pass
                    else:
                        queue_item.scoring_request.total_wait_time += wait_time

                self.__request_metrics.add_result(
                    request_id=queue_item.scoring_request.internal_id,
                    response_code=e.status_code,
                    response_payload=e.response_payload,
                    model_response_code=e.model_response_code,
                    model_response_reason=e.model_response_reason,
                    additional_wait_time=wait_time,
                    request_total_wait_time=queue_item.scoring_request.total_wait_time)

                self._log_score_failed_with_retriable_exception(queue_item.scoring_request, back_off, wait_time)

                await asyncio.sleep(back_off)

                queue_item.scoring_request.scoring_duration += end - start
                self.__scoring_request_queue.append(queue_item)

            except Exception:
                end = time.time()

                self._log_score_failed_with_unhandled_exception(queue_item.scoring_request, traceback.format_exc())
                await asyncio.sleep(1)

                queue_item.scoring_request.scoring_duration += end - start
                self.__scoring_request_queue.append(queue_item)

        self.__is_running = False

        if self._configuration.async_mode:
            lu.get_logger().debug("Worker {}: Finished processing an queue item from mini-batch {}"
                                  .format(self.id, mini_batch_id))
            set_mini_batch_id(None)
        else:
            lu.get_logger().debug("Worker {}: Finished".format(self.id))

    def _request_exceeds_max_retry_time_interval(self, scoring_request: ScoringRequest) -> bool:
        if self._configuration.max_retry_time_interval is None:
            return False

        return scoring_request.scoring_duration > self._configuration.max_retry_time_interval

    def _log_score_failed_with_max_retry_time_interval_exceeded(self, scoring_request: ScoringRequest):
        msg = "Worker {}: Score failed: Payload scored for {} seconds, " \
              + "which exceeded the maximum time interval of {} seconds. " \
              + "internal_id: {}, total_wait_time: {}, retry_count: {}"
        lu.get_logger().error(msg.format(self.id,
                                         round(scoring_request.scoring_duration, 2),
                                         self._configuration.max_retry_time_interval,
                                         scoring_request.internal_id,
                                         scoring_request.total_wait_time,
                                         scoring_request.retry_count))

    def _log_score_failed_with_permanent_exception(self,
                                                   scoring_request: ScoringRequest,
                                                   exception: PermanentException):
        msg = "Worker {}: Score failed: {}. internal_id: {}, total_wait_time: {}, retry_count: {}"
        lu.get_logger().error(msg.format(
            self.id,
            str(exception),
            scoring_request.internal_id,
            scoring_request.total_wait_time,
            scoring_request.retry_count))

    def _log_score_failed_with_retriable_exception(self,
                                                   scoring_request: ScoringRequest,
                                                   back_off: int,
                                                   wait_time: int):
        msg = "Worker {}: Encountered retriable exception. internal_id: {},"
        msg += " back_off: {}, wait_time: {}, total_wait_time: {}, retry_count: {}"
        lu.get_logger().debug(msg.format(
            self.id,
            scoring_request.internal_id,
            back_off,
            wait_time,
            scoring_request.total_wait_time,
            scoring_request.retry_count))

    def _log_score_failed_with_unhandled_exception(self, scoring_request: ScoringRequest, trace: str):
        lu.get_logger().error("Worker {}: Encountered unhandled exception. internal_id: {}, error: {}".format(
            self.id,
            scoring_request.internal_id,
            trace))

    async def __process_queue_item(self, queue_item: QueueItem):
        if queue_item.timeout_generator is None:
            queue_item.timeout_generator = ScoringClient.get_retry_timeout_generator(self.__client_session.timeout)

        timeout = ScoringClient.get_next_retry_timeout(queue_item.timeout_generator)

        if self._configuration.segment_large_requests == "enabled":
            if queue_item.segmented_score_context is None:
                queue_item.segmented_score_context = SegmentedScoreContext(queue_item.scoring_request,
                                                                           self._configuration.segment_max_token_size)

            if queue_item.segmented_score_context.has_more():

                scoring_result = await queue_item.segmented_score_context.score_next_once(
                    self.__scoring_client,
                    self.__client_session,
                    timeout,
                    worker_id=self.id)

                if scoring_result.status == ScoringResultStatus.SUCCESS:
                    if queue_item.segmented_score_context.has_more():
                        # Reset timeout generator for the next segment.
                        queue_item.timeout_generator = ScoringClient.get_retry_timeout_generator(
                            self.__client_session.timeout)
                        self.__scoring_request_queue.append(queue_item)
                    else:
                        scoring_result = queue_item.segmented_score_context.build_scoring_result(scoring_result)
                        self.__add_result(queue_item.scoring_request, scoring_result)
                else:
                    self.__add_result(queue_item.scoring_request, scoring_result)

        else:
            scoring_result = await self.__scoring_client.score_once(
                session=self.__client_session,
                scoring_request=queue_item.scoring_request,
                timeout=timeout,
                worker_id=self.id)

            self.__add_result(queue_item.scoring_request, scoring_result)

    def __add_result(self, scoring_request: ScoringRequest, scoring_result: ScoringResult):
        self.__scoring_result_queue.append(scoring_result)

        self.__request_metrics.add_result(
            scoring_request.internal_id,
            scoring_result.status,
            "<OMITTED>" if scoring_result.status == ScoringResultStatus.SUCCESS else scoring_result.response_body,
            "" if not scoring_result.response_headers else scoring_result.response_headers.get(
                "ms-azureml-model-error-statuscode", ""),
            "" if not scoring_result.response_headers else scoring_result.response_headers.get(
                "ms-azureml-model-error-reason", ""),
            0,
            scoring_request.total_wait_time)

        lu.get_events_client().emit_row_completed(1, str(scoring_result.status))
