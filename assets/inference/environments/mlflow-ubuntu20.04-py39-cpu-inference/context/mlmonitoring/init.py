"""For init."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=global-statement
from .logger import init_logging, logger
from .config import init_config, teardown_config, get_config
from .queue import init_queue, teardown_queue, get_queue
from .worker import init_worker, teardown_worker

sdk_ready = False


def is_sdk_ready():
    """For is sdk ready."""
    global sdk_ready
    return sdk_ready


def init(model_version=None):
    """For init."""
    global sdk_ready
    if sdk_ready:
        return

    init_logging()

    logger.info("init data collector")

    # init configuration for MDC
    init_config(model_version)

    config = get_config()
    if config.is_debug():
        logger.warning("data collector debugging is enabled")

    if not config.enabled():
        logger.warning("data collector is not enabled")
        return

    # create default payload queue in memory
    init_queue(config.queue_capacity())
    logger.debug("data collector in-memory queue created")

    # start worker thread for payload sending
    queue = get_queue()
    if not config.worker_disabled():
        init_worker(queue, config)
        logger.debug("data collector worker started")
    else:
        logger.warning("data collector worker is disabled")

    # init done
    logger.info("data collector ready")
    sdk_ready = True


def teardown(wait_for_flush=False):
    """For teardown."""
    global sdk_ready
    if not sdk_ready:
        return

    logger.debug("tear down data collector")
    teardown_worker(wait_for_flush)
    teardown_queue()
    teardown_config()

    # tear down done
    sdk_ready = False
