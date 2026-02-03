"""For config."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# pylint: disable=global-statement
import os
import json

from ..logger import is_debug, logger

default_queue_capacity = 1500
default_worker_count = 1
default_sample_rate_percentage = 100


class MdcConfig:
    """For MdcConfig."""

    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            enabled=False,
            host="127.0.0.1",
            port=50011,
            debug=False,
            sample_rate_percentage=default_sample_rate_percentage,
            model_version=None,
            queue_capacity=default_queue_capacity,
            worker_disabled=False,
            worker_count=default_worker_count,
            local_capture=False,
            compact_format=False,
    ):
        """For init."""
        self._debug = debug
        self._enabled = enabled
        self._sample_rate_percentage = sample_rate_percentage
        self._host = host
        self._port = port
        self._model_version = model_version
        # queue - max length
        self._queue_capacity = queue_capacity
        # worker - disabled for test purpose only
        self._worker_disabled = worker_disabled
        self._worker_count = worker_count
        # payload sender
        self._local_capture = local_capture
        self._compact_format = compact_format
        self._collections = {}

    def is_debug(self):
        """For is debug."""
        return self._debug

    def enabled(self):
        """For enabled."""
        return self._enabled

    def set_enabled(self, enabled):
        """For set enabled."""
        self._enabled = enabled

    def compact_format(self):
        """For compact format."""
        return self._compact_format

    def sample_rate_percentage(self):
        """For sample rate percentage."""
        return self._sample_rate_percentage

    def host(self):
        """For host."""
        return self._host

    def port(self):
        """For port."""
        return self._port

    def model_version(self):
        """For model version."""
        return self._model_version

    def queue_capacity(self):
        """For queue capacity."""
        return self._queue_capacity

    def worker_disabled(self):
        """For worker disabled."""
        return self._worker_disabled

    def worker_count(self):
        """For worker count."""
        return self._worker_count

    def local_capture(self):
        """For local capture."""
        return self._local_capture

    def add_collection(self, col_name, enabled=False, sample_rate_percentage=100):
        """For add collection."""
        self._collections[col_name] = {
            "enabled": enabled,
            "sampleRatePercentage": sample_rate_percentage,
        }

    def collections(self):
        """For collections."""
        return self._collections

    def collection_enabled(self, collection_name):
        """For collection enabled."""
        path = os.getenv("AZUREML_MDC_CONFIG_PATH")
        if not path:
            # for legacy settings, we depend on a global switch to see whether collections are enabled or not.
            return self.enabled()

        for n, c in self._collections.items():
            if n == collection_name:
                return c.get("enabled", False)

        return False

    def collection_sample_rate_percentage(self, collection_name):
        """For collection sample rate percentage."""
        path = os.getenv("AZUREML_MDC_CONFIG_PATH")
        if not path:
            # for legacy settings, we take the global sample_rate_percentage.
            return self.sample_rate_percentage()

        for n, c in self._collections.items():
            if n == collection_name:
                return c.get("sampleRatePercentage", default_sample_rate_percentage)

        return default_sample_rate_percentage


def loadConfig(model_version=None):
    """For loadConfig."""
    debug = is_debug()
    path = os.getenv("AZUREML_MDC_CONFIG_PATH")

    if path:
        with open(path) as f:
            cfg = json.load(f)
            mdc_cfg = MdcConfig(
                host=os.getenv("AZUREML_MDC_HOST", "127.0.0.1"),
                port=int(os.getenv("AZUREML_MDC_PORT", "50011")),
                debug=debug,
                model_version=model_version,
                local_capture=cfg.get("runMode", "cloud") == "local"
            )

            collection_cfg = cfg.get("collections", {})
            custom_logging_enabled = False
            for col_name, c in collection_cfg.items():
                col_name_lower = col_name.lower()
                if c.get("enabled", False):
                    mdc_cfg.add_collection(col_name_lower, True, c.get("sampleRatePercentage", 100))
                    if col_name_lower not in ('request', 'response'):
                        custom_logging_enabled = True

            mdc_cfg.set_enabled(custom_logging_enabled)

            return mdc_cfg

    enabled = os.getenv("AZUREML_MDC_ENABLED", "false")
    if enabled.lower() == "true":
        return MdcConfig(
            enabled=True,
            host=os.getenv("AZUREML_MDC_HOST", "127.0.0.1"),
            port=int(os.getenv("AZUREML_MDC_PORT", "50011")),
            debug=debug,
            sample_rate_percentage=int(os.getenv("AZUREML_MDC_SAMPLE_RATE", str(default_sample_rate_percentage))),
            queue_capacity=int(os.getenv("AZUREML_MDC_QUEUE_CAPACITY", str(default_queue_capacity))),
            worker_disabled=os.getenv("AZUREML_MDC_WORKER_DISABLED", "false").lower() == "true",
            worker_count=int(os.getenv("AZUREML_MDC_WORKER_COUNT", str(default_worker_count))),
            compact_format=os.getenv("AZUREML_MDC_FORMAT_COMPACT", "false").lower() == "true",
            local_capture=os.getenv("AZUREML_MDC_LOCAL_CAPTURE", "false").lower() == "true",
            model_version=model_version,
        )

    return MdcConfig(enabled=False, debug=debug)


mdc_config = None


def init_config(model_version=None):
    """For init config."""
    global mdc_config
    mdc_config = loadConfig(model_version)

    logger.info("mdc enabled: %r", mdc_config.enabled())
    logger.info("mdc collections count %d", len(mdc_config.collections()))
    for n, c in mdc_config.collections().items():
        logger.info("mdc collection %s <enabled:%r,sample_percentage:%d>",
                    n,
                    c.get("enabled", False),
                    c.get("sampleRatePercentage", default_sample_rate_percentage))

    if mdc_config.is_debug():
        config_json = json.dumps(mdc_config.__dict__)
        logger.debug("mdc config: %s", config_json)


def teardown_config():
    """For teardown config."""
    global mdc_config
    mdc_config = None


def get_config():
    """For get config."""
    global mdc_config
    return mdc_config
