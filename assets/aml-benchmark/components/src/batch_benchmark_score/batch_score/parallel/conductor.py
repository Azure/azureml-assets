# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for conductor."""

import asyncio
import os
import aiohttp

from aiohttp import TraceConfig
from collections import deque
from ..utils import logging_utils as lu
from ..utils.logging_utils import get_events_client
from ..utils.scoring_result import ScoringResult
from ..utils.scoring_client import ScoringClient
from ..utils.scoring_request import ScoringRequest
from .adjustment import AIMD
from .request_metrics import RequestMetrics
from .worker import Worker, QueueItem


class Conductor:
    """The conductor class."""

    def __init__(
            self,
            loop: asyncio.AbstractEventLoop,
            scoring_client: ScoringClient,
            segment_large_requests: str,
            segment_max_token_size: int,
            initial_worker_count: int,
            max_worker_count: int,
            max_retry_time_interval: int = None,
            trace_configs: "list[TraceConfig]" = None):
        """Init method."""
        self.__loop: asyncio.AbstractEventLoop = loop
        self.__scoring_client: ScoringClient = scoring_client
        self.__segment_large_requests: str = segment_large_requests
        self.__segment_max_token_size: int = segment_max_token_size
        self.__initial_worker_count: int = initial_worker_count
        self.__max_worker_count: int = max_worker_count
        self.__max_retry_time_interval = max_retry_time_interval

        self.__trace_configs = trace_configs

        if self.__max_worker_count is not None:
            self.__target_worker_count = min(self.__initial_worker_count, self.__max_worker_count)
        else:
            self.__target_worker_count = self.__initial_worker_count

        self.__client_session: aiohttp.ClientSession = None
        self.__workers: list[Worker] = []
        self.__scoring_request_queue: deque = deque()
        self.__result_list: list[ScoringResult] = []
        self.__request_metrics = RequestMetrics()
        self.__cas = AIMD(request_metrics=self.__request_metrics)

    def close_session(self):
        """Close session."""
        if self.__client_session is not None and not self.__client_session.closed:
            lu.get_logger().info("Conductor: Closing ClientSession")

            self.__loop.run_until_complete(self.__client_session.close())

            # As per https://docs.aiohttp.org/en/stable/client_advanced.html#graceful-shutdown
            # NOTE: Should be fixed in aiohttp 4.0.0, at time of writing aiohttp is on 3.8.3
            # "The appropriate amount of time to wait will vary from application to application"
            self.__loop.run_until_complete(asyncio.sleep(1))

        else:
            lu.get_logger().info("Conductor: No open ClientSession exists")

    async def run(self, requests: "list[ScoringRequest]") -> "list[ScoringResult]":
        """Run method."""
        self.__add_requests(requests)

        lu.get_logger().info(
            "Conductor: Starting with {} running workers and {} target worker count. "
            "There are {} workers in the worker pool.".format(
                len(list(filter(lambda worker: worker.is_running, self.__workers))),
                self.__target_worker_count, len(self.__workers)))

        client_session = await self.__get_session()
        tasks: "list[asyncio.Task]" = []

        target_result_len = len(self.__scoring_request_queue)

        while len(self.__result_list) < target_result_len:

            adjustments = self.__cas.calculate_next_concurrency(self.__target_worker_count)

            if adjustments.new_concurrency <= self.__target_worker_count or \
                    (adjustments.new_concurrency > self.__target_worker_count and
                     adjustments.new_concurrency < len(self.__scoring_request_queue)):

                lu.get_logger().info(
                    "Conductor: Taking adjustment of {} to {}".format(
                        self.__target_worker_count, adjustments.new_concurrency))
                self.__target_worker_count = adjustments.new_concurrency

                if self.__max_worker_count is not None:
                    if self.__max_worker_count < self.__target_worker_count:
                        lu.get_logger().debug(
                            "Conductor: Overriding with self.__max_worker_count value of {}".format(
                                self.__max_worker_count))
                    # Upper bound is self.__max_worker_count, if present
                    self.__target_worker_count = min(self.__target_worker_count, self.__max_worker_count)

            get_events_client().emit_worker_concurrency(self.__target_worker_count)
            lu.get_logger().info("Conductor: __target_worker_count set to {}".format(self.__target_worker_count))

            while len(self.__workers) < self.__target_worker_count:
                worker = Worker(
                    scoring_client=self.__scoring_client,
                    client_session=client_session,
                    scoring_request_queue=self.__scoring_request_queue,
                    result_list=self.__result_list,
                    request_metrics=self.__request_metrics,
                    segment_large_requests=self.__segment_large_requests,
                    segment_max_token_size=self.__segment_max_token_size,
                    id=str(len(self.__workers)),
                    max_retry_time_interval=self.__max_retry_time_interval)

                self.__workers.append(worker)

            for i, worker in enumerate(self.__workers):
                if i < self.__target_worker_count and not worker.is_running():
                    tasks.append(self.__loop.create_task(worker.start()))
                elif i >= self.__target_worker_count and worker.is_running():
                    worker.stop()

            lu.get_logger().debug("Conductor: Conductor sleeping {} seconds".format(adjustments.next_adjustment_delay))
            tasks = await self.__sleep(adjustments.next_adjustment_delay, tasks, target_result_len)

        for task in tasks:
            task.cancel()

        for worker in self.__workers:
            worker.stop()

        ret: list[ScoringResult] = []

        # Remove items from __result_list without creating a new reference
        while len(self.__result_list) > 0:
            scoring_result = self.__result_list.pop()
            if not scoring_result.omit:
                ret.append(scoring_result)

        return ret

    def __add_requests(self, requests: "list[ScoringRequest]"):
        for request in requests:
            self.__scoring_request_queue.append(QueueItem(scoring_request=request))

    async def __get_session(self) -> aiohttp.ClientSession:
        if self.__client_session is None:
            total = 30 * 60  # Default timeout to 30 minutes
            if self.__segment_large_requests == "enabled":
                total = self.__segment_max_token_size * 1  # Assume 1 second per token generation
                total = max(total, 3 * 60)  # Lower bound of 3 minutes

            timeout_override = os.environ.get("BATCH_SCORE_REQUEST_TIMEOUT")
            total = int(timeout_override) if timeout_override else total

            lu.get_logger().info(f"Setting client-side request timeout to {total} seconds.")
            timeout = aiohttp.ClientTimeout(total=total)
            self.__client_session = aiohttp.ClientSession(
                timeout=timeout,
                trace_configs=self.__trace_configs
            )

        return self.__client_session

    async def __sleep(
            self, duration: int,  tasks: "list[asyncio.Task]", target_result_len: int) -> "list[asyncio.Task]":
        """Wait for the configured sleep interval, but wakes up early if the workers finish."""
        sleep_task = asyncio.create_task(asyncio.sleep(duration))
        tasks = [*tasks, sleep_task]

        while not sleep_task.done() and len(self.__result_list) < target_result_len:
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        sleep_task.cancel()

        return list(tasks)
