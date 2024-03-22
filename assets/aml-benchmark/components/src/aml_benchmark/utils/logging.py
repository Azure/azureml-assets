# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains helper functions for logging."""

from typing import Any, Dict
from importlib.metadata import entry_points
from azureml.telemetry.logging_handler import get_appinsights_log_handler
from azureml.telemetry import INSTRUMENTATION_KEY
from azureml.exceptions import AzureMLException
from azureml._common._error_definition.azureml_error import AzureMLError  # type: ignore
from .constants import TelemetryConstants, AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT
from .constants import ExceptionTypes, KNOWN_MODULES
from azureml.core import Run
from azureml.core.run import _OfflineRun


import logging
import os
import hashlib
import mlflow
import platform
import re



class DummyWorkspace:
    """Dummy Workspace class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "local-ws"
        self.subscription_id = ""
        self.location = "local"

    def get_mlflow_tracking_uri(self):
        """Mlflow Tracking URI.

        Returns:
            _type_: _description_
        """
        return mlflow.get_tracking_uri()


class DummyExperiment:
    """Dummy Experiment class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "offline_default_experiment"
        self.id = "1"
        self.workspace = DummyWorkspace()


class TestRun:
    """Main class containing Current Run's details."""

    def __init__(self):
        """__init__."""
        self._run = Run.get_context()
        if isinstance(self._run, _OfflineRun):
            self._experiment = DummyExperiment()
            self._workspace = self._experiment.workspace
        else:
            self._experiment = self._run.experiment
            self._workspace = self._experiment.workspace

    @property
    def run(self):
        """Azureml Run.

        Returns:
            _type_: _description_
        """
        return self._run

    @property
    def experiment(self):
        """Azureml Experiment.

        Returns:
            _type_: _description_
        """
        return self._experiment

    @property
    def workspace(self):
        """Azureml Workspace.

        Returns:
            _type_: _description_
        """
        return self._workspace

    @property
    def compute(self):
        """Azureml compute instance.

        Returns:
            _type_: _description_
        """
        if not isinstance(self._run, _OfflineRun):
            target_name = self._run.get_details()["target"]
            if self.workspace.compute_targets.get(target_name):
                return self.workspace.compute_targets[target_name].vm_size
            else:
                return "serverless"
        return "local"

    @property
    def region(self):
        """Azure Region.

        Returns:
            _type_: _description_
        """
        return self._workspace.location

    @property
    def subscription(self):
        """Azureml Subscription.

        Returns:
            _type_: _description_
        """
        return self._workspace.subscription_id

    @property
    def parent_run(self):
        """Get Root run of the pipeline.

        Returns:
            _type_: _description_
        """
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun) or (cur_run.parent is None):
            return self._run
        if cur_run.parent is not None:
            cur_run = cur_run.parent
        return cur_run

    @property
    def root_run(self):
        """Get Root run of the pipeline.

        Returns:
            _type_: _description_
        """
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun) or (cur_run.parent is None):
            return self._run
        while cur_run.parent is not None:
            cur_run = cur_run.parent
        return cur_run

    @property
    def root_attribute(self):
        """Get Root attribute of the pipeline.

        Returns:
            _type_: str
        """
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun):
            return cur_run.id
        cur_attribute = cur_run.name
        first_parent = cur_run.parent
        if first_parent is not None and hasattr(first_parent, "parent"):
            second_parent = first_parent.parent
            if second_parent is not None and hasattr(second_parent, "name"):
                cur_attribute = second_parent.name
        return cur_attribute

    @property
    def get_extra_run_info(self):
        """Get run details of the pipeline.

        Returns:
            _type_: _description_
        """
        info = {}
        if not isinstance(self._run, _OfflineRun):
            raw_json = self._run.get_details()
            if raw_json["runDefinition"]["inputAssets"].get("mlflow_model", None) is not None:
                try:
                    model_asset_id = raw_json["runDefinition"]["inputAssets"]["mlflow_model"]["asset"]["assetId"]
                    info["model_asset_id"] = model_asset_id
                    if model_asset_id.startswith("azureml://registries"):
                        info["model_source"] = "registry"
                        model_info = re.search("azureml://registries/(.+)/models/(.+)/versions/(.+)",
                                               model_asset_id)
                        info["model_registry_name"] = model_info.group(1)
                        info["model_name"] = model_info.group(2)
                        info["model_version"] = model_info.group(3)
                    else:
                        info["model_source"] = "workspace"
                except Exception:
                    pass
            try:
                module_name = raw_json['properties'].get('azureml.moduleName', 'Unknown')
                info["moduleName"] = module_name if module_name in KNOWN_MODULES else 'Unknown'
                if info["moduleName"] != 'Unknown':
                    module_id = raw_json['properties'].get('azureml.moduleid', '')
                    info['moduleId'] = module_id
                    if module_id.startswith("azureml://registries"):
                        info["moduleSource"] = "registry"
                        module_info = re.search("azureml://registries/(.+)/components/(.+)/versions/(.+)",
                                                module_id)
                        info["moduleRegistryName"] = module_info.group(1)
                        info["moduleVersion"] = module_info.group(3)
                    else:
                        info["moduleSource"] = "workspace"
                        info["moduleVersion"] = raw_json['properties'].get('azureml.moduleVersion', 'Unknown')
                info["pipeline_type"] = raw_json['properties'].get('PipelineType', None)
                info["source"] = raw_json['properties'].get('Source', None)
            except Exception:
                pass
        try:
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT")
            location = re.compile("//(.*?)\\.").search(location).group(1)
        except Exception:
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT", "")
        info["location"] = location
        return info




class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(self) -> None:
        """__init__.

        Args:
            run_details (_type_, optional): _description_. Defaults to None.
            app_name (_type_, optional): _description_. Defaults to TelemetryConstants.COMPONENT_NAME.
            model_evaluation_version : _description_. Defaults to TelemetryConstants.COMPONENT_DEFAULT_VERSION.
            os_info (_type_, optional): _description_. Defaults to platform.system().
            task_type (str, optional): _description_. Defaults to "".
        """
        run_details = TestRun()
        self.app_name = TelemetryConstants.DEFAULT_APP_NAME
        self.run_id = run_details.run.id
        self.compute_target = run_details.compute
        self.experiment_id = run_details.experiment.id
        self.parent_run_id = run_details.parent_run.id
        self.root_run_id = run_details.root_run.id
        self.os_info = platform.system()
        self.region = run_details.region
        self.subscription_id = run_details.subscription
        self.rootAttribution = run_details.root_attribute
        run_info = run_details.get_extra_run_info
        self.location = run_info.get("location", "")

        self.moduleVersion = run_info.get("moduleVersion", TelemetryConstants.DEFAULT_VERSION)
        if run_info.get("moduleId", None):
            self.moduleId = run_info.get("moduleId")
        if run_info.get("moduleSource", None):
            self.moduleSource = run_info.get("moduleSource")
        if run_info.get("moduleRegistryName", None):
            self.moduleRegistryName = run_info.get("moduleRegistryName")

        if run_info.get("model_asset_id", None):
            self.model_asset_id = run_info.get("model_asset_id")
        if run_info.get("model_source", None):
            self.model_source = run_info.get("model_source")
        if run_info.get("model_registry_name", None):
            self.model_registry_name = run_info.get("model_registry_name")
        if run_info.get("model_name", None):
            self.model_name = run_info.get("model_name")
        if run_info.get("model_version", None):
            self.model_version = run_info.get("model_version")

        if run_info.get("pipeline_type", None):
            self.pipeline_type = run_info.get("pipeline_type")
        if run_info.get("source", None):
            self.source = run_info.get("source")
    
    @property
    def dict(self):
        """Convert object to dict.

        Returns:
            _type_: dict
        """
        return vars(self)

custom_dimensions = CustomDimensions().dict
    


class AppInsightsPIIStrippingFormatter(logging.Formatter):
    """Formatter for App Insights Logging.

    Args:
        logging (_type_): _description_
    """

    def format(self, record: logging.LogRecord) -> str:
        """Format incoming log record.

        Args:
            record (logging.LogRecord): _description_

        Returns:
            str: _description_
        """
        exception_tb = getattr(record, 'exception_tb_obj', None)
        if exception_tb is None:
            return super().format(record)

        not_available_message = '[Not available]'

        properties = getattr(record, 'properties', {})

        message = properties.get('exception_message', TelemetryConstants.NON_PII_MESSAGE)
        traceback_msg = properties.get('exception_traceback', not_available_message)

        record.message = record.msg = '\n'.join([
            'Type: {}'.format(properties.get('error_type', ExceptionTypes.Unclassified)),
            'Class: {}'.format(properties.get('exception_class', not_available_message)),
            'Message: {}'.format(message),
            'Traceback: {}'.format(traceback_msg),
            'ExceptionTarget: {}'.format(properties.get('exception_target', not_available_message))
        ])

        # Update exception message and traceback in extra properties as well
        properties['exception_message'] = message

        return super().format(record)


def log_mlflow_params(**kwargs: Any) -> None:
    """
    Log the provided key-value pairs as parameters in MLflow.

    If a file path or a list of file paths is provided in the value, the checksum of the
    file(s) is calculated and logged as the parameter value. If `None` is provided as
    the value, the parameter is not logged.

    :param **kwargs: Key-value pairs of parameters to be logged in MLflow.
    :return: None
    """
    MLFLOW_PARAM_VALUE_MAX_LEN = 500
    OVERFLOW_STR = '...'
    params = {}
    for key, value in kwargs.items():
        if isinstance(value, str) and os.path.isfile(value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(open(value, "rb").read()).hexdigest()
            params[key] = checksum
        elif isinstance(value, list) and all(isinstance(item, str) and os.path.isfile(item) for item in value):
            # calculate checksum of input dataset
            checksum = hashlib.sha256(b"".join(open(item, "rb").read() for item in value)).hexdigest()
            params[key] = checksum
        else:
            if value is not None:
                if isinstance(value, str) and len(value) > MLFLOW_PARAM_VALUE_MAX_LEN:
                    value_len = MLFLOW_PARAM_VALUE_MAX_LEN - len(OVERFLOW_STR)
                    params[key] = value[: value_len] + OVERFLOW_STR
                else:
                    params[key] = value

    mlflow.log_params(params)


def get_logger(filename: str) -> logging.Logger:
    """
    Create and configure a logger based on the provided filename.

    This function creates a logger with the specified filename and configures it
    by setting the logging level to INFO, adding a StreamHandler to the logger,
    and specifying a specific log message format.

    :param filename: The name of the file associated with the logger.
    :return: The configured logger.
    """
    logger = logging.getLogger(filename)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    logger.addHandler(stream_handler)

    try:
        for custom_logger in entry_points(group=AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT):
            handler = custom_logger.load()
            logger.addHandler(handler())
    except TypeError:
        # For Older python versions
        custom_loggers = entry_points().get(AML_BENCHMARK_DYNAMIC_LOGGER_ENTRY_POINT, ())
        for custom_logger in custom_loggers:
            if custom_logger is not None:
                handler = custom_logger.load()
                logger.addHandler(handler())

    formatter = logging.Formatter(
        "[%(asctime)s - %(name)s - %(levelname)s] - %(message)s"
    )
    stream_handler.setFormatter(formatter)

    handler_names = [handler.get_name() for handler in logger.handlers]

    if TelemetryConstants.APP_INSIGHT_HANDLER_NAME not in handler_names:
        child_namespace = __name__
        current_logger = logging.getLogger("azureml.telemetry").getChild(child_namespace)
        current_logger.propagate = False
        current_logger.setLevel(logging.CRITICAL)
        appinsights_handler = get_appinsights_log_handler(
            instrumentation_key=INSTRUMENTATION_KEY,
            logger=current_logger, properties=custom_dimensions
        )
        formatter = AppInsightsPIIStrippingFormatter(
            fmt='%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s \
                [%(process)d] %(message)s \n'.format(TelemetryConstants.DEFAULT_APP_NAME,
                                                     custom_dimensions.get("run_id", ""))
        )
        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(logging.INFO)
        appinsights_handler.set_name(TelemetryConstants.APP_INSIGHT_HANDLER_NAME)
        logger.addHandler(appinsights_handler)
    return logger


logger = get_logger(__name__)


def log_params_and_metrics(
    parameters: Dict[str, Any],
    metrics: Dict[str, Any],
    log_to_parent: bool,
) -> None:
    """Log mlflow params and metrics to current run and parent run."""
    filtered_metrics = {}
    for key in metrics:
        if isinstance(metrics[key], bool):
            # For bool value, latest version of mlflow throws an error.
            filtered_metrics[key] = float(metrics[key])
        elif isinstance(metrics[key], (int, float)):
            filtered_metrics[key] = metrics[key]
    # Log to current run
    logger.info(
        f"Attempting to log {len(parameters)} parameters and {len(filtered_metrics)} metrics."
    )
    try:
        log_mlflow_params(**parameters)
    except Exception as ex:
        logger.error(f"Failed to log parameters to current run due to {ex}")
    try:
        mlflow.log_metrics(filtered_metrics)
    except Exception as ex:
        logger.error(f"Failed to log metrics to current run due to {ex}")
    if log_to_parent:
        # Log to parent run
        try:
            parent_run_id = Run.get_context().parent.id
            ml_client = mlflow.tracking.MlflowClient()
            for param_name, param_value in parameters.items():
                param_value_to_log = param_value
                if isinstance(param_value, str) and len(param_value) > 500:
                    param_value_to_log = param_value[: 497] + '...'
                try:
                    ml_client.log_param(parent_run_id, param_name, param_value_to_log)
                except Exception as ex:
                    logger.error(f"Failed to log parameter {param_name} to root run due to {ex}.")
            for metric_name, metric_value in filtered_metrics.items():
                try:
                    ml_client.log_metric(parent_run_id, metric_name, metric_value)
                except Exception as ex:
                    logger.error(f"Failed to log metric {metric_name} to root run due to {ex}.")
        except Exception as ex:
            logger.error(f"Failed to log parameters and metrics to root run due to {ex}.")
