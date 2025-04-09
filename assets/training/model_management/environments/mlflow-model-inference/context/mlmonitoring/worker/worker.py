"""For worker."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=global-statement
import logging
from threading import Thread
import traceback

from .sender import create_sender


class MdcWorker:
    """For MdcWorker."""

    def __init__(self, queue, config):
        """For init."""
        self.logger = logging.getLogger("mdc.worker")
        self._queue = queue
        self._config = config
        self._stopped = True
        self._threads = []
        self._sender = create_sender(config)

    def start(self):
        """For start."""
        self._stopped = False
        # create worker threads
        for index in range(self._config.worker_count()):
            thread = Thread(target=self._thread_run, kwargs={'index': index}, daemon=True)
            self._threads.append(thread)
        # start all
        for thread in self._threads:
            thread.start()

    def enqueue(self, payload):
        """For enqueue."""
        return self._queue.enqueue(payload)

    def stop(self, wait_for_flush=False):
        """For stop."""
        if not wait_for_flush:
            self._stopped = True

        # must stop the queue, or threads may keep waiting on empty queue without exit.
        self._queue.close()

        for thread in self._threads:
            thread.join()

        self._threads = []
        self.logger.debug("data collector worker stopped")

    def _thread_run(self, index):
        """For thread run."""
        self.logger.debug("worker thread %d: start", index)
        while not self._stopped:
            payload, queue_len = self._queue.dequeue()
            if payload is None:  # queue closed
                break

            # processing the payload
            self.logger.debug("worker thread %d: sending payload %s@%s, queue length - %d",
                              index, payload.designation(), payload.id(), queue_len)
            try:
                succeed, msg = self._sender.send(payload)
                if not succeed:
                    self.logger.error("worker thread %d: send payload %s@%s failed - %s",
                                      index, payload.designation(), payload.id(), msg)
                else:
                    self.logger.debug("worker thread %d: send payload %s@%s succeeded - %s",
                                      index, payload.designation(), payload.id(), msg)
            except TypeError as err:
                traceback.print_exc()
                self.logger.error("worker thread %d: send payload %s@%s raise exception: %s",
                                  index, payload.designation(), payload.id(), "{0}".format(err))

        self.logger.debug("worker thread %d: exit", index)


mdc_worker = None


def init_worker(queue, config):
    """For init worker."""
    global mdc_worker
    mdc_worker = MdcWorker(queue, config)
    # start worker thread
    mdc_worker.start()


def get_worker():
    """For get worker."""
    global mdc_worker
    return mdc_worker


def teardown_worker(wait_for_flush=False):
    """For teardown worker."""
    global mdc_worker
    if mdc_worker is not None:
        mdc_worker.stop(wait_for_flush)
        mdc_worker = None
