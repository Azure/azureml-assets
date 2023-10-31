# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for Worker."""

import asyncio
import aiohttp
import traceback
import os
import time

from collections import deque
from ..utils.common.common import str2bool
from ..utils import logging_utils as lu
from ..utils.scoring_result import RetriableException, ScoringResult, ScoringResultStatus
from ..utils.scoring_client import ScoringClient
from ..utils.scoring_request import ScoringRequest
from ..utils.segmented_score_context import SegmentedScoreContext
from .request_metrics import RequestMetrics


class QueueItem:
    """Class for queue item."""

    def __init__(
            self,
            scoring_request: ScoringRequest, segmented_score_context: SegmentedScoreContext = None
    ):
        """Init for queue item."""
        self.scoring_request = scoring_request
        self.segmented_score_context = segmented_score_context


class Worker:
    """Class for worker."""

    def __init__(
            self,
            scoring_client: ScoringClient,
            client_session: aiohttp.ClientSession,
            scoring_request_queue: "deque[QueueItem]",
            result_list: "list[ScoringResult]",
            request_metrics: RequestMetrics,
            segment_large_requests: str,
            segment_max_token_size: int,
            id: str,
            max_retry_time_interval: int = None):
        """Init for worker."""
        self.__scoring_client: ScoringClient = scoring_client
        self.__client_session = client_session

        self.id = id
        self.__is_running = False

        self.__scoring_request_queue = scoring_request_queue
        self.__result_list = result_list
        self.__request_metrics = request_metrics

        self.__segment_max_token_size: int = segment_max_token_size
        self.__segment_large_requests: str = segment_large_requests

        self.__enable_delay_after_success = str2bool(
            os.environ.get("BATCH_SCORE_DELAY_AFTER_SUCCESSFUL_REQUEST", "True"))
        self.__max_retry_time_interval = max_retry_time_interval

        lu.get_logger().debug("{}: Created".format(self.id))

    def is_running(self) -> bool:
        """Is worker running."""
        return self.__is_running

    def stop(self):
        """Stop worker."""
        lu.get_logger().debug("{}: Stopped".format(self.id))
        self.__is_running = False

    async def start(self):
        """Start worker."""
        self.__is_running = True
        lu.set_worker_id(self.id)
        lu.get_logger().debug("{}: Started".format(self.id))

        while len(self.__scoring_request_queue) > 0 and self.__is_running:
            queue_item = self.__scoring_request_queue.pop()

            start, end = 0, 0

            # If the request's scoring duration exceeds the limit,
            # log as failure and don't add it back to the queue
            if self.__max_retry_time_interval is not None and \
                    queue_item.scoring_request.scoring_duration > self.__max_retry_time_interval:

                result = ScoringResult(
                    status=ScoringResultStatus.FAILURE,
                    start=0,
                    end=0,
                    request_obj=queue_item.scoring_request.original_payload_obj,
                    request_metadata=queue_item.scoring_request.request_metadata,
                    response_body=None,
                    response_headers=None,
                    num_retries=queue_item.scoring_request.retry_count,
                    omit=False
                )

                self.__add_result(queue_item.scoring_request, result)

                lu.get_logger().debug(
                    "{}: Score failed: Payload scored for {} seconds, "
                    "which exceeded the maximum time interval of {} seconds. internal_id: {}, "
                    "total_wait_time: {}, retry_count: {}".format(
                        self.id,
                        round(queue_item.scoring_request.scoring_duration, 2),
                        self.__max_retry_time_interval,
                        queue_item.scoring_request.internal_id,
                        queue_item.scoring_request.total_wait_time,
                        queue_item.scoring_request.retry_count))
            else:
                try:
                    start = time.time()
                    await self.__process_queue_item(queue_item)
                    end = time.time()

                    if self.__enable_delay_after_success:
                        # On successful request, add a 1 second delay before processing the next one
                        # This is to lower the chance of other requests getting starved because
                        # a worker that releases
                        # quota always immediately re-reserve it with no delay.
                        await asyncio.sleep(1)

                except RetriableException as e:
                    end = time.time()

                    # Back-off 1 second before adding ScoringRequest back to the queue
                    # TODO: consider adjusting back-off.
                    #   - None-429s: increase backoff for consecutive retries
                    #   - 429s: vary backoff based on request cost to prevent large request starvation
                    back_off = e.retry_after or 1
                    queue_item.scoring_request.retry_count += 1
                    wait_time = 0

                    if e.status_code == 429 or e.model_response_code == "429":
                        wait_time = back_off
                        queue_item.scoring_request.total_wait_time += wait_time

                    self.__request_metrics.add_result(
                        request_id=queue_item.scoring_request.internal_id,
                        response_code=e.status_code,
                        response_payload=e.response_payload,
                        model_response_code=e.model_response_code,
                        model_response_reason=e.model_response_reason,
                        additional_wait_time=wait_time,
                        request_total_wait_time=queue_item.scoring_request.total_wait_time)

                    lu.get_logger().debug(
                        "{}: Encountered retriable exception. internal_id: {}, "
                        "back_off: {}, wait_time: {}, total_wait_time: {}, retry_count: {}".format(
                            self.id,
                            queue_item.scoring_request.internal_id,
                            back_off,
                            wait_time,
                            queue_item.scoring_request.total_wait_time,
                            queue_item.scoring_request.retry_count))

                    await asyncio.sleep(back_off)

                    queue_item.scoring_request.scoring_duration += end - start
                    self.__scoring_request_queue.append(queue_item)

                except Exception as e:
                    print(e)
                    end = time.time()

                    lu.get_logger().error(
                        "{}: Encountered unhandled exception. internal_id: {}, error: {}".format(
                            self.id, queue_item.scoring_request.internal_id, traceback.format_exc()))
                    await asyncio.sleep(1)

                    queue_item.scoring_request.scoring_duration += end - start
                    self.__scoring_request_queue.append(queue_item)

        self.__is_running = False
        lu.get_logger().debug("{}: Finished".format(self.id))

    async def __process_queue_item(self, queue_item: QueueItem):
        if self.__segment_large_requests == "enabled":
            if queue_item.segmented_score_context is None:
                queue_item.segmented_score_context = SegmentedScoreContext(
                    queue_item.scoring_request, self.__segment_max_token_size)

            if queue_item.segmented_score_context.has_more():
                scoring_result = await queue_item.segmented_score_context.score_next_once(
                    self.__scoring_client, self.__client_session, worker_id=self.id)

                if scoring_result.status == ScoringResultStatus.SUCCESS:
                    if queue_item.segmented_score_context.has_more():
                        self.__scoring_request_queue.append(queue_item)
                    else:
                        scoring_result = queue_item.segmented_score_context.build_scoring_result(
                            scoring_result)
                        self.__add_result(queue_item.scoring_request, scoring_result)
                else:
                    self.__add_result(queue_item.scoring_request, scoring_result)

        else:
            scoring_result = await self.__scoring_client.score_once(
                session=self.__client_session,
                scoring_request=queue_item.scoring_request,
                worker_id=self.id)

            self.__add_result(queue_item.scoring_request, scoring_result)

    def __add_result(self, scoring_request: ScoringRequest, scoring_result: ScoringResult):
        self.__result_list.append(scoring_result)

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
