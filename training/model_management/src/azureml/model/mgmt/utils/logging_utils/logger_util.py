# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
This file defines the util functions used for logging
"""

import logging
import uuid
from . import run_utils
from .config import Config
from azureml.telemetry import get_telemetry_log_handler
from azureml.automl.core.shared.telemetry_formatter import (
    AppInsightsPIIStrippingFormatter,
)
import codecs
from typing import Any

CODEC = "base64"
INSTRUMENTATION_KEY = b"NzFiOTU0YTgtNmI3ZC00M2Y1LTk4NmMtM2QzYTY2MDVkODAz"


def _get_custom_dimension():
    """
    gets custom dimensions like subscition_id, workspace name, region, etc. used by the logger
    :return dictionary containing the custom dimensions
    """

    custom_dim = {
        "sub_id": run_utils._get_sub_id(),
        "ws_name": run_utils._get_ws_name(),
        "region": run_utils._get_region(),
        "run_id": run_utils._get_run_id(),
        "compute_target_name": run_utils._get_compute(),
        "compute_target_type": run_utils._get_compute_vm_size(),
    }
    return custom_dim


class GllmHandler(logging.StreamHandler):
    """
    GLLM Handler class extended from logging.StreamHandler, used as handler for the logger
    """

    def emit(self, record: logging.LogRecord) -> None:
        """
        Emits logs to stream after adding custom dimensions
        """

        new_properties = getattr(record, "properties", {})
        new_properties.update({"log_id": str(uuid.uuid4())})
        custom_dimensions = _get_custom_dimension()
        cust_dim_copy = custom_dimensions.copy()
        cust_dim_copy.update(new_properties)
        setattr(record, "properties", cust_dim_copy)
        msg = self.format(record)
        if record.levelname == "ERROR" and "AzureMLException" not in record.message:
            setattr(
                record,
                "exception_tb_obj",
                "non-azureml exception raised so scrubbing",
            )
        stream = self.stream
        stream.write(msg)


def get_logger_app(
    logging_level: str = "DEBUG",
    custom_dimensions: dict = None,
    name: str = Config.LOGGER_NAME,
):
    """
    Creates handlers and define formatter for emitting logs to AppInsights
    Also adds handlers to HF logs
    :returns logger which emits logs to stdOut and appInsights with PII Scrubbing
    """

    numeric_log_level = getattr(logging, logging_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError("Invalid log level: %s" % logging_level)

    logger = logging.getLogger(name)

    # don't log twice i.e. root logger
    logger.propagate = False

    logger.setLevel(numeric_log_level)

    handler_names = [handler.get_name() for handler in logger.handlers]

    run_id = run_utils._get_run_id()
    app_name = Config.FINETUNE_APP_NAME

    if Config.AMLFT_HANDLER_NAME not in handler_names:
        # create Gllm handler and set formatter
        format_str = (
            "%(asctime)s [{}] [{}] [%(module)s] %(funcName)s "
            "%(lineno)s: %(levelname)-8s [%(process)d] %(message)s \n"
        )
        formatter = logging.Formatter(format_str.format(app_name, run_id))
        stream_handler = GllmHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_log_level)
        stream_handler.set_name(Config.AMLFT_HANDLER_NAME)
        logger.addHandler(stream_handler)

    if Config.APP_INSIGHT_HANDLER_NAME not in handler_names:
        # create AppInsight handler and set formatter
        instrumentation_key = instrumentation_key or codecs.decode(INSTRUMENTATION_KEY, CODEC)
        appinsights_handler = get_telemetry_log_handler(
            instrumentation_key=Config.INSTRUMENTATION_KEY_AML_OLD,
            component_name="automl",
        )
        formatter = AppInsightsPIIStrippingFormatter(
            fmt=(
                "%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d]"
                " %(message)s \n".format(app_name, run_id)
            )
        )
        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(numeric_log_level)
        appinsights_handler.set_name(Config.APP_INSIGHT_HANDLER_NAME)
        logger.addHandler(appinsights_handler)

    return logger
