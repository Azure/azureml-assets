# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging utils."""

from azureml.model.mgmt.config import AppName, LoggerConfig
from azureml.model.mgmt.utils.run_utils import JobRunDetails
from azureml.telemetry import get_telemetry_log_handler
from azureml.telemetry._telemetry_formatter import ExceptionFormatter
import platform
import uuid
import codecs
import logging
import sys


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(
        self,
        run_details,
        app_name=AppName.IMPORT_MODEL,
    ) -> None:
        """Init Custom dimensions."""
        # run_details
        self.run_id = run_details.run_id
        self.parent_run_id = run_details.parent_run_id
        self.subscription_id = run_details.subscription_id
        self.workspace_name = run_details.workspace_name
        self.root_attribution = run_details.root_attribute
        self.region = run_details.region
        self.experiment_id = run_details.experiment_id
        self.compute_target = run_details.compute
        self.vm_size = run_details.vm_size
        self.component_asset_id = run_details.component_asset_id

        # component execution info
        self.os_info = platform.system()
        self.app_name = app_name

        self._add_model_download_args()
        self._add_model_preprocess_args()

    def update_custom_dimensions(self, properties: dict):
        """Add/update properties in custom dimensions."""
        assert isinstance(properties, dict)
        self.__dict__.update(properties)

    def _add_model_download_args(self):
        args = sys.argv
        if "--model-id" in args:
            ind = args.index("--model-id")
            self.model_id = sys.argv[ind + 1]

        if "--model-source" in args:
            ind = args.index("--model-source")
            self.model_source = sys.argv[ind + 1]

    def _add_model_preprocess_args(self):
        args = sys.argv
        if "--model-id" in args:
            ind = args.index("--model-id")
            self.model_id = sys.argv[ind + 1]

        if "--task-name" in args:
            ind = args.index("--task-name")
            self.task_name = sys.argv[ind + 1]

        if "--mlflow-flavor" in args:
            ind = args.index("--mlflow-flavor")
            self.mlflow_flavor = sys.argv[ind + 1]


class ModelImportHandler(logging.StreamHandler):
    """Model Import handler for stream handling."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emit logs to stream after adding custom dimensions."""
        new_properties = getattr(record, "properties", {})
        new_properties.update({'log_id': str(uuid.uuid4())})
        custom_dims_dict = custom_dimensions.__dict__
        cust_dim_copy = custom_dims_dict.copy()
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


def get_logger(name=LoggerConfig.LOGGER_NAME, level=LoggerConfig.VERBOSITY_LEVEL):
    """Return logger with adding necessary handlers."""
    logger = logging.getLogger(name)

    numeric_log_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError("Invalid log level: %s" % level)

    # don't log twice i.e. root logger
    logger.propagate = False
    logger.setLevel(numeric_log_level)
    handler_names = [handler.get_name() for handler in logger.handlers]
    app_name = LoggerConfig.LOGGER_NAME

    if LoggerConfig.MODEL_IMPORT_HANDLER_NAME not in handler_names:
        format_str = "%(asctime)s [%(module)s] %(funcName)s: %(levelname)-8s %(message)s \n"
        formatter = logging.Formatter(format_str.format(app_name, run_details.run_id))
        stream_handler = ModelImportHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(numeric_log_level)
        stream_handler.set_name(LoggerConfig.MODEL_IMPORT_HANDLER_NAME)
        logger.addHandler(stream_handler)

    if LoggerConfig.APPINSIGHT_HANDLER_NAME not in handler_names:
        instrumentation_key = codecs.decode(LoggerConfig.INSTRUMENTATION_KEY, LoggerConfig.CODEC).decode("utf-8")

        appinsights_handler = get_telemetry_log_handler(
            instrumentation_key=instrumentation_key,
            component_name="automl",
        )

        formatter = ExceptionFormatter(
            fmt="%(asctime)s [{}] [{}] [%(module)s] %(funcName)s: %(levelname)-8s %(message)s\n".format(
                app_name, run_details.run_id
            )
        )
        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(numeric_log_level)
        appinsights_handler.set_name(LoggerConfig.APPINSIGHT_HANDLER_NAME)
        logger.addHandler(appinsights_handler)

    return logger


run_details = JobRunDetails.get_run_details()
custom_dimensions = CustomDimensions(run_details=run_details)
