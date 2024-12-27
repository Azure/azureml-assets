# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging utilities for the evaluator."""

from azure.ml.component.run import CoreRun
import platform

import logging

from constants import TelemetryConstants, ExceptionTypes


class DummyWorkspace:
    """Dummy Workspace class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "local-ws"
        self.subscription_id = ""
        self.location = "local"
        self.resource_group = ""


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
        self._run = CoreRun.get_context()
        if not hasattr(self._run, 'experiment'):
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
        if hasattr(self._run, 'experiment'):
            target_name = self._run.get_details()["target"]
            try:
                if self.workspace.compute_targets.get(target_name):
                    return self.workspace.compute_targets[target_name].vm_size
                else:
                    return "serverless"
            except Exception:
                return "Unknown"
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
        if not hasattr(cur_run, 'experiment') or (cur_run.parent is None):
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
        if not hasattr(cur_run, 'experiment') or (cur_run.parent is None):
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
        if not hasattr(cur_run, 'experiment'):
            return cur_run.id
        cur_attribute = cur_run.name
        first_parent = cur_run.parent
        if first_parent is not None and hasattr(first_parent, "parent"):
            second_parent = first_parent.parent
            if second_parent is not None and hasattr(second_parent, "name"):
                cur_attribute = second_parent.name
        return cur_attribute


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(self,
                 run_details,
                 app_name="online_eval",
                 component_version="0.0.1",
                 os_info=platform.system()) -> None:
        """__init__.

        Args:
            run_details (_type_, optional): _description_. Defaults to None.
            app_name (_type_, optional): _description_. Defaults to TelemetryConstants.COMPONENT_NAME.
            component_version : _description_. Defaults to TelemetryConstants.COMPONENT_DEFAULT_VERSION.
            os_info (_type_, optional): _description_. Defaults to platform.system().
            task_type (str, optional): _description_. Defaults to "".
        """
        self.app_name = app_name
        self.run_id = run_details.run.id
        self.compute_target = run_details.compute
        self.experiment_id = run_details.experiment.id
        self.parent_run_id = run_details.parent_run.id
        self.root_run_id = run_details.root_run.id
        self.os_info = os_info
        self.region = run_details.region
        self.subscription_id = run_details.subscription
        self.rootAttribution = run_details.root_attribute
        self.moduleVersion = component_version


current_run = TestRun()
custom_dimensions = CustomDimensions(current_run)


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
            'ExceptionTarget: {}'.format(properties.get('exception_target', not_available_message)),
        ])
        record.msg += " | Properties: {}"

        # Update exception message and traceback in extra properties as well
        properties['exception_message'] = message
        record.properties = properties
        record1 = super().format(record)
        return record1


class CustomLogRecord(logging.LogRecord):
    """Custom Log Record class for App Insights."""

    def __init__(self, *args, **kwargs):
        """__init__."""
        super().__init__(*args, **kwargs)
        self.properties = getattr(self, "properties", {})


# Step 2: Set the custom LogRecord factory
def custom_log_record_factory(*args, **kwargs):
    """Get CustomLogRecord for App Insights."""
    return CustomLogRecord(*args, **kwargs)


def get_logger(logging_level: str = 'INFO',
               custom_dimensions: dict = vars(custom_dimensions),
               name: str = "online_eval"):
    """Get logger.

    Args:
        logging_level (str, optional): _description_. Defaults to 'DEBUG'.
        custom_dimensions (dict, optional): _description_. Defaults to vars(custom_dimensions).
        name (str, optional): _description_. Defaults to TelemetryConstants.LOGGER_NAME.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    numeric_log_level = getattr(logging, logging_level.upper(), None)
    if not isinstance(numeric_log_level, int):
        raise ValueError('Invalid log level: %s' % logging_level)

    logger = logging.getLogger(name)
    logger.propagate = True
    logger.setLevel(numeric_log_level)
    logging.setLogRecordFactory(custom_log_record_factory)
    handler_names = [handler.get_name() for handler in logger.handlers]
    run_id = custom_dimensions["run_id"]

    if TelemetryConstants.APP_INSIGHT_HANDLER_NAME not in handler_names:
        try:
            from azure.ai.ml._telemetry.logging_handler import AzureMLSDKLogHandler, INSTRUMENTATION_KEY
            from azure.ai.ml._user_agent import USER_AGENT
            custom_properties = {"PythonVersion": platform.python_version()}
            custom_properties.update({"user_agent": USER_AGENT})
            custom_properties.update(custom_dimensions)
            appinsights_handler = AzureMLSDKLogHandler(
                connection_string=f"InstrumentationKey={INSTRUMENTATION_KEY}",
                custom_properties=custom_properties,
                enable_telemetry=True
            )
            formatter = AppInsightsPIIStrippingFormatter(
                fmt='%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s \
                    [%(process)d] %(message)s \n'.format("online_eval", run_id)
            )
            appinsights_handler.setFormatter(formatter)
            appinsights_handler.setLevel(numeric_log_level)
            appinsights_handler.set_name(TelemetryConstants.APP_INSIGHT_HANDLER_NAME)
            logger.addHandler(appinsights_handler)
        except Exception as e:
            logger.warning(f"Failed to add App Insights handler: {e}")

    return logger
