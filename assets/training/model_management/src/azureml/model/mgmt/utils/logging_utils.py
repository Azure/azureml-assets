# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging utils."""

from azureml.core import Run
from azureml.core.compute import ComputeTarget
from azureml.model.mgmt.config import AppName, LoggerConfig
from azureml.telemetry import get_telemetry_log_handler
from azureml.automl.core.shared.telemetry_formatter import (
    AppInsightsPIIStrippingFormatter,
)
import platform
import uuid
import codecs
import logging
import sys


class RunDetails:
    def __init__(self):
        self._run = Run.get_context()

    @property
    def run_id(self):
        """RunID of existing run."""
        return self._run.id

    @property
    def parent_run_id(self):
        """Parent RunID of existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._run.parent.id

    def workspace(self):
        """Return workspace."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._run.experiment.workspace

    @property
    def workspace_name(self):
        """Return workspace."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self.workspace.name

    @property
    def experiment_id(self):
        """Return SubID."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._run.experiment.id

    @property
    def subscription_id(self):
        """Return SubID."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self.workspace.subscription_id

    @property
    def region(self):
        """Return the region where the run is executing."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self.workspace.location

    @property
    def compute(self):
        """Return compute target for current run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        details = self._run.get_details()
        return details.get("target", "")

    @property
    def vm_size(self):
        """Return compute VM size."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        compute_name = self.compute
        if compute_name == "":
            return "No compute found."
        # TODO: Use V2 way of determining this.
        cpu_cluster = ComputeTarget(workspace=self._run.experiment.workspace, name=compute_name)
        return cpu_cluster.vm_size

    @property
    def root_attribute(self):
        """Root attribute of run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE

        cur_attribute = self._run.name
        run = self._run.parent
        # update current run's root_attribute to the root run.
        while run != None:
            cur_attribute = run.name
            run = run.parent
        return cur_attribute

    def __str__(self):
        return {
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "subscription_id": self.subscription_id,
            "workspace_name": self.workspace_name,
            "root_attribute": self.root_attribute,
            "experiment_id": self.experiment_id,
            "region": self.region,
            "compute": self.compute,
            "vm_size": self.vm_size,
        }


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(
        self,
        run_details,
        app_name=AppName.IMPORT_MODEL,
    ) -> None:
        # run_details
        self.run_id = run_details.run_id
        self.parent_run_id = run_details.parent_run_id
        self.subscription_id = run_details.subscription_id
        self.workspace_name = run_details.workspace_name
        self.rootAttribution = run_details.root_attribute
        self.region = run_details.region
        self.experiment_id = run_details.experiment_id
        self.compute_target = run_details.compute
        self.vm_size = run_details.vm_size

        # component execution info
        self.os_info = platform.system()
        self.app_name = app_name
        self.import_model_version = LoggerConfig.IMPORT_MODEL_VERSION

        self._add_model_download_args()
        self._add_model_preprocess_args()

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
        """Emits logs to stream after adding custom dimensions."""
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
        format_str = (
            "%(asctime)s [{}] [{}] [%(module)s] %(funcName)s "
            "%(lineno)s: %(levelname)-8s [%(process)d] %(message)s \n"
        )
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

        formatter = AppInsightsPIIStrippingFormatter(
            fmt=(
                "%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d]"
                " %(message)s \n".format(app_name, run_details.run_id)
            )
        )
        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(numeric_log_level)
        appinsights_handler.set_name(LoggerConfig.APPINSIGHT_HANDLER_NAME)
        logger.addHandler(appinsights_handler)

    return logger


run_details = RunDetails()
custom_dimensions = CustomDimensions(run_details=run_details)
