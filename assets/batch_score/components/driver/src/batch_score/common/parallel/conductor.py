import asyncio
import os
from collections import deque

import aiohttp
from aiohttp import TraceConfig

from ...batch_pool.routing.routing_client import RoutingClient
from ...batch_pool.scoring.scoring_client import ScoringClient
from ..configuration.client_settings import NullClientSettingsProvider
from ..configuration.configuration import Configuration
from ..post_processing.gatherer import Gatherer
from ..post_processing.mini_batch_context import MiniBatchContext
from ..scoring.scoring_request import ScoringRequest
from ..scoring.scoring_result import ScoringResult
from ..telemetry import logging_utils as lu
from ..telemetry.events import event_utils
from ..telemetry.logging_utils import get_events_client
from .adjustment import AIMD
from .request_metrics import RequestMetrics
from .worker import QueueItem, Worker


class Conductor:
    def __init__(
        self, 
        configuration: Configuration,
        loop: asyncio.AbstractEventLoop,
        scoring_client: ScoringClient,
        routing_client: RoutingClient,
        trace_configs: "list[TraceConfig]" = None,
        finished_callback = None,
    ):
        self._configuration = configuration
        self.__loop: asyncio.AbstractEventLoop = loop
        self.__scoring_client: ScoringClient = scoring_client

        self.__trace_configs = trace_configs

        if self._configuration.max_worker_count is not None:
            self.__target_worker_count = min(self._configuration.initial_worker_count, self._configuration.max_worker_count)
        else:
            self.__target_worker_count = self._configuration.initial_worker_count

        self.__client_session: aiohttp.ClientSession = self.__get_session()
        self.__routing_client: RoutingClient = routing_client
        self.__workers: list[Worker] = []
        self.__scoring_request_queue: deque = deque()
        self.__scoring_result_queue: deque = deque()
        self.__request_metrics = RequestMetrics()
        self.__tasks: list[asyncio.Task] = []

        if self._configuration.async_mode:
            self.__finished_callback = finished_callback
            self.__minibatch_index_set = set()
            self.__failed_scoring_result_queue: deque = deque()

        # When the target is a single endpoint, the routing client is not present.
        # Then default to NullClientSettingsProvider.
        self.__clients_settings_provider = self.__routing_client or NullClientSettingsProvider()
        self.__cas = AIMD(request_metrics=self.__request_metrics, client_settings_provider=self.__clients_settings_provider)

        if self._configuration.async_mode:
            self._init_workers()
            self.__loop.run_in_executor(None, self.__loop.run_forever)

    async def run(self, requests: "list[ScoringRequest]") -> "list[ScoringResult]":
        self.__add_requests(requests)

        lu.get_logger().info("Conductor: Starting with {} running workers and {} target worker count. There are {} workers in the worker pool.".format(len(list(filter(lambda worker: worker.is_running, self.__workers))), self.__target_worker_count, len(self.__workers)))

        target_result_len = len(self.__scoring_request_queue)

        while len(self.__scoring_result_queue) < target_result_len:
            next_adjustment_delay = self._update_target_worker_count()
            self._adjust_worker_concurrency()
            lu.get_logger().debug("Conductor: Conductor sleeping {} seconds".format(next_adjustment_delay))
            self.__tasks = await self.__sleep(next_adjustment_delay, self.__tasks, target_result_len)

        await self._release()

        ret = [result for result in self.__scoring_result_queue if not result.omit]

        # Clear the scoring result queue for the next minibatch to be run.
        # We cannot reset using `self.__scoring_result_queue = []` because this list is referenced by the workers.
        self.__scoring_result_queue.clear()

        return ret
    
    def enqueue(self, requests: "list[ScoringRequest]", failed_results: "list[ScoringResult]", mini_batch_context: MiniBatchContext):
        self.__minibatch_index_set.add(mini_batch_context.mini_batch_id)
        if len(requests) == 0:
            lu.get_logger().info("Conductor: Encountered empty requests in minibatch id {}, adding empty results.".format(mini_batch_context.mini_batch_id))
            self.__gatherer.add_empty_result(mini_batch_context)
            lu.get_events_client().emit_mini_batch_completed(
                input_row_count=mini_batch_context.target_result_len,
                output_row_count=0)
            event_utils.generate_minibatch_summary(
                minibatch_id=mini_batch_context.minibatch_index,
                output_row_count=0,
            )
            return
        
        self.__add_requests(requests)
        self.__add_failed_scoring_results(failed_results)

        lu.get_logger().info("Conductor: Enqueued {} scoring requests. ".format(len(requests)))

    def check_all_tasks_processed(self):
        lu.get_logger().info("Conductor: Checking if all tasks are processed, ")
        lu.get_logger().info("Conductor: len minibatch_index_set is {}, self.__gatherer.get_returned_minibatch_count() is {}".format(len(self.__minibatch_index_set), self.__gatherer.get_returned_minibatch_count()))
        # no input items, no output items, and minibatch set count is equal to minibatch return count
        return len(self.__scoring_request_queue) == 0 and len(self.__scoring_result_queue) == 0 and len(self.__minibatch_index_set) == self.__gatherer.get_returned_minibatch_count()

    def get_finished_batch_result(self):
        return self.__gatherer.get_finished_minibatch_result()
    
    def get_processing_batch_number(self):
        lu.get_logger().info("Conductor: Getting number of processing mini batches.")
        lu.get_logger().info("Conductor: len minibatch_index_set is {}, self.__gatherer.get_returned_minibatch_count() is {}".format(len(self.__minibatch_index_set), self.__gatherer.get_returned_minibatch_count()))
        return len(self.__minibatch_index_set) - self.__gatherer.get_returned_minibatch_count()

    def shutdown(self):
        if self._configuration.async_mode:
            asyncio.run(self._release())
        
        self._close_session()
    
    def __add_requests(self, requests: "list[ScoringRequest]"):
        for request in requests:
            timeout_generator = ScoringClient.get_retry_timeout_generator(self.__client_session.timeout)
            self.__scoring_request_queue.append(QueueItem(scoring_request=request, timeout_generator=timeout_generator))

    def __add_failed_scoring_results(self, failed_results: "list[ScoringResult]"):
        self.__failed_scoring_result_queue.extend(failed_results)

    def __configure_session_timeout(self) -> aiohttp.ClientTimeout:
        # Max retry interval is configurable by user
        if self._configuration.max_retry_time_interval:
            total = self._configuration.max_retry_time_interval

        # Default when segmentation enabled
        elif self._configuration.segment_large_requests == "enabled":
            total = self._configuration.segment_max_token_size * 1  # Assume 1 second per token generation
            total = max(total, 3 * 60)  # Lower bound of 3 minutes

        # Default when segmentation disabled
        else:
            total = 30 * 60  # Default timeout of 30 minutes

        # Environment override takes top priority but may be improperly set. In that case, use the fallback total.
        timeout_override = os.environ.get("BATCH_SCORE_REQUEST_TIMEOUT")
        if timeout_override:
            try:
                total = int(timeout_override)
            except ValueError:
                lu.get_logger().warning(f"Invalid value for BATCH_SCORE_REQUEST_TIMEOUT: {timeout_override}")

        lu.get_logger().info(f"Setting client-side request timeout to {total} seconds.")
        return aiohttp.ClientTimeout(total=total)

    def __get_session(self) -> aiohttp.ClientSession:
        timeout = self.__configure_session_timeout()
        client_session = aiohttp.ClientSession(
            timeout=timeout,
            trace_configs=self.__trace_configs,
            loop = self.__loop)
        
        return client_session

    async def __sleep(self, duration: int,  tasks: "list[asyncio.Task]", target_result_len: int = None) -> "list[asyncio.Task]":
        """Waits for the configured sleep interval, but wakes up early if the workers finish."""

        sleep_task = asyncio.create_task(asyncio.sleep(duration))
        tasks = [*tasks, sleep_task]

        while not sleep_task.done() and (target_result_len is None or len(self.__scoring_result_queue) < target_result_len):
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        sleep_task.cancel()

        return list(tasks)

    def _adjust_worker_concurrency(self):
        '''Adjusts the number of running workers to match the target worker count.'''
        while len(self.__workers) < self.__target_worker_count:
            worker = Worker(
                configuration=self._configuration,
                scoring_client=self.__scoring_client,
                client_session=self.__client_session,
                client_settings_provider=self.__clients_settings_provider,
                scoring_request_queue=self.__scoring_request_queue,
                scoring_result_queue=self.__scoring_result_queue,
                request_metrics=self.__request_metrics,
                id=str(len(self.__workers)),
            )

            self.__workers.append(worker)

        for i, worker in enumerate(self.__workers):
            if i < self.__target_worker_count and not worker.is_running():
                self.__tasks.append(self.__loop.create_task(worker.start()))
            elif i >= self.__target_worker_count and worker.is_running():
                worker.stop()

    def _update_target_worker_count(self):
        '''Update the target worker count and returns the amount of time to sleep before the next adjustment.'''
        adjustments = self.__cas.calculate_next_concurrency(self.__target_worker_count)

        if adjustments.new_concurrency <= self.__target_worker_count or \
                (adjustments.new_concurrency > self.__target_worker_count and adjustments.new_concurrency < len(self.__scoring_request_queue)):
            lu.get_logger().info("Conductor: Taking adjustment of target worker count {} to {}".format(self.__target_worker_count, adjustments.new_concurrency))
            self.__target_worker_count = adjustments.new_concurrency

            if self._configuration.max_worker_count is not None:
                if self._configuration.max_worker_count < self.__target_worker_count:
                    lu.get_logger().debug("Conductor: Overriding with self._configuration.max_worker_count value of {}".format(self._configuration.max_worker_count))
                    # Upper bound is self._configuration.max_worker_count, if present
                self.__target_worker_count = min(self.__target_worker_count, self._configuration.max_worker_count)

        get_events_client().emit_worker_concurrency(self.__target_worker_count)
        lu.get_logger().info("Conductor: __target_worker_count set to {}".format(self.__target_worker_count))
        return adjustments.next_adjustment_delay
    
    def _init_workers(self):
        lu.get_logger().debug("Conductor: Initializing {} workers".format(self._configuration.initial_worker_count))

        # Start regular workers
        self._adjust_worker_concurrency()
        lu.get_logger().info("Conductor: Starting with {} running workers and {} target worker count. There are {} workers in the worker pool."
                             .format(len(list(filter(lambda worker: worker.is_running, self.__workers))), self.__target_worker_count, len(self.__workers)))

        # Start specialized workers
        self._init_gatherer()
        self._init_monitor()

    def _init_gatherer(self):
        self.__gatherer = Gatherer(self.__scoring_result_queue, self.__failed_scoring_result_queue, self.__finished_callback)
        self.__tasks.append(self.__loop.create_task(self.__gatherer.run()))

    def _init_monitor(self):
        self.__tasks.append(self.__loop.create_task(self._monitor_run()))

    async def _monitor_run(self):
        while True:
            next_adjustment_delay = self._update_target_worker_count()
            self._adjust_worker_concurrency()
            lu.get_logger().debug("Conductor: Conductor sleeping {} seconds".format(next_adjustment_delay))

            self.__tasks = await self.__sleep(next_adjustment_delay, self.__tasks)

    async def _release(self):
        for task in self.__tasks:
            task.cancel()
            
        for worker in self.__workers:
            worker.stop()

    def _close_session(self):
        if self.__client_session is not None and not self.__client_session.closed:
            lu.get_logger().info("Conductor: Closing ClientSession")

            asyncio.run(self.__client_session.close())

            # As per https://docs.aiohttp.org/en/stable/client_advanced.html#graceful-shutdown
            # NOTE: Should be fixed in aiohttp 4.0.0, at time of writing aiohttp is on 3.8.3
            # "The appropriate amount of time to wait will vary from application to application"
            asyncio.sleep(1)

        else:
            lu.get_logger().info("Conductor: No open ClientSession exists")
