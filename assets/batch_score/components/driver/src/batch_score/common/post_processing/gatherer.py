# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import asyncio
import copy
from collections import defaultdict, deque

from ..scoring.scoring_result import ScoringResult
from ..telemetry import logging_utils as lu
from ..telemetry.logging_utils import set_mini_batch_id


class Gatherer:
    class Metrics:
        def __init__(self):
            self.handled_scoring_result_count = 0
            self.handled_minibatch_count = 0
            self.returned_minibatch_count = 0

        def emit_scoring_result_handled(self):
            self.handled_scoring_result_count += 1

        def emit_minibatch_handled(self):
            self.handled_minibatch_count += 1

        def emit_minibatch_returned(self, minibatch_count):
            self.returned_minibatch_count += minibatch_count

    def __init__(
            self,
            scoring_result_queue: "deque[ScoringResult]",
            failed_scoring_result_queue: "deque[ScoringResult]",
            callback):
        self.__finished_callback = callback
        self.__scoring_result_queue = scoring_result_queue
        self.__failed_scoring_result_queue = failed_scoring_result_queue

        self.__finished_minibatch_id_set = set()
        self.__metrics = Gatherer.Metrics()
        self.__result_list_map = defaultdict(list)
        self.__working = True

        self.__finished_result_list_map = {}

    async def run(self):
        while self.__working:
            mini_batch_id = None
            try:
                if self.__failed_scoring_result_queue:
                    scoring_result = self.__failed_scoring_result_queue.popleft()
                elif self.__scoring_result_queue:
                    scoring_result = self.__scoring_result_queue.popleft()
                else:
                    await asyncio.sleep(1)
                    continue

                mini_batch_id = scoring_result.mini_batch_context.mini_batch_id
                set_mini_batch_id(mini_batch_id)

                if mini_batch_id in self.__finished_minibatch_id_set:
                    lu.get_logger().info(f"Gatherer: Received scoring_result, mini_batch_id : {mini_batch_id}, "
                                         "but already finished, omit")
                    continue

                self.__result_list_map[mini_batch_id].append(scoring_result)
                if len(self.__result_list_map[mini_batch_id]) == scoring_result.mini_batch_context.target_result_len:
                    self._move_minibatch_to_finished_result_list_queue(scoring_result)
                self.__metrics.emit_scoring_result_handled()
            except Exception as e:
                if mini_batch_id not in self.__finished_minibatch_id_set:
                    if mini_batch_id is not None:
                        '''already get a item, this exception could bind with a mini batch result'''
                        scoring_result.mini_batch_context.exception = e
                        self._add_to_finished_result_list_map([], scoring_result.mini_batch_context, exception=e)
                    else:
                        lu.get_logger().error("Gatherer: Received 'none' mini batch id in scoring result. "
                                              f"Exception: {e}")
                        lu.get_logger().error("Gatherer: Skipping the scoring result. Current unfinished "
                                              f"mini batch count: {len(self.__result_list_map)}")

        set_mini_batch_id(None)
        lu.get_logger().info("Gatherer: Received None, exiting")
        return

    def _move_minibatch_to_finished_result_list_queue(self, scoring_result):
        mini_batch_context = scoring_result.mini_batch_context
        mini_batch_id = mini_batch_context.mini_batch_id
        lu.get_logger().debug("Gatherer: Move Result: {}, target_result_len : {}"
                              .format(
                                  len(self.__result_list_map[mini_batch_id]),
                                  mini_batch_context.target_result_len))

        ret = [result for result in self.__result_list_map[mini_batch_id] if not result.omit]
        result_after_callback = self.__finished_callback(ret, mini_batch_context)
        self._add_to_finished_result_list_map(result_after_callback, mini_batch_context)

    def _add_to_finished_result_list_map(self, result_list, mini_batch_context, exception=None):
        self.__finished_result_list_map[mini_batch_context.mini_batch_id] = {
            "ret": result_list,
            "mini_batch_context": mini_batch_context.raw_mini_batch_context,
            "exception": exception
        }
        self.__finished_minibatch_id_set.add(mini_batch_context.mini_batch_id)
        if mini_batch_context.mini_batch_id in self.__result_list_map:
            del self.__result_list_map[mini_batch_context.mini_batch_id]
        self.__metrics.emit_minibatch_handled()

    def add_empty_result(self, mini_batch_context):
        self._add_to_finished_result_list_map([], mini_batch_context)

    def get_finished_minibatch_result(self) -> "dict[str, dict[str, any]]":
        result_map = copy.deepcopy(self.__finished_result_list_map)
        lu.get_logger().info(f"Gatherer: get_finished_minibatch_result. Mini batch IDs: {list(result_map.keys())}")
        self.__finished_result_list_map.clear()
        self.__metrics.emit_minibatch_returned(len(result_map))
        return result_map

    def get_returned_minibatch_count(self):
        return self.__metrics.returned_minibatch_count
