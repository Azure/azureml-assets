from azureml.core import Run
from azureml.core.compute import ComputeTarget
from azureml.telemetry import get_telemetry_log_handler
from azureml.automl.core.shared.telemetry_formatter import (
    AppInsightsPIIStrippingFormatter,
)
import platform
import uuid
import codecs
import logging
from .config import Config


CODEC = "base64"
INSTRUMENTATION_KEY = b"NzFiOTU0YTgtNmI3ZC00M2Y1LTk4NmMtM2QzYTY2MDVkODAz"
MODEL_IMPORT_HANDLER_NAME = "ModelImportHandler" 
APPINSIGHT_HANDLER_NAME = "AppInsightsHandler"
COMPONENT_NAME = "ModelImport"
IMPORT_MODEL_VERSION = "0.0.5" ## Update when changing version in spec file.


class RunDetails():
    def __init__(self):
        self._run = Run.get_context()

    @property
    def id(self):
        """RunID of existing run."""
        if "OfflineRun" in self._run.id:
            return self._run.id
        return self._run.parent.id

    @property
    def subscription_id(self):
        """Return SubID."""
        if "OfflineRun" in self._run.id:
            return Config.OFFLINE_RUN_MESSAGE
        return self._run.experiment.workspace.subscription_id

    @property
    def workspace(self):
        """Return workspace name."""
        if "OfflineRun" in self._run.id:
            return Config.OFFLINE_RUN_MESSAGE
        return self._run.experiment.workspace.name

    @property
    def region(self):
        """Return the region where the run is executing."""
        if "OfflineRun" in self._run.id:
            return Config.OFFLINE_RUN_MESSAGE
        return self._run.experiment.workspace.location

    @property
    def compute(self):
        """Return compute target for current run."""
        if "OfflineRun" in self._run.id:
            return Config.OFFLINE_RUN_MESSAGE
        details = self._run.get_details()
        return details.get("target", "")

    @property
    def vm_size(self):
        """Return compute VM size."""
        if "OfflineRun" in self._run.id:
            return Config.OFFLINE_RUN_MESSAGE
        compute_name = self.compute
        if compute_name == "":
            return "No compute found."
        # Use V2 way of determining this.
        cpu_cluster = ComputeTarget(workspace=self._run.experiment.workspace, name=compute_name)
        return cpu_cluster.vm_size

    @property
    def root_attribute(self):
        """Root attribute of run."""
        cur_attribute = self._run.name
        run = self._run.parent
        # update current run's root_attribute to the root run.
        while run != None:
            cur_attribute = run.name
            run = run.parent
        return cur_attribute


class CustomDimensions:
    """Custom Dimensions Class for App Insights."""

    def __init__(self,
                 app_name=COMPONENT_NAME,
                 run_id=None,
                 compute_target=None,
                 experiment_id=None,
                 parent_run_id=None,
                 os_info=platform.system(),
                 region=None,
                 subscription_id=None,
                 task_type="",
                 root_attribute="local") -> None:
        self.app_name = app_name
        self.run_id = run_id
        self.model_evaluation_version = IMPORT_MODEL_VERSION
        self.compute_target = compute_target
        self.experiment_id = experiment_id
        self.parent_run_id = parent_run_id
        self.os_info = os_info
        self.region = region
        self.subscription_id = subscription_id
        self.task_type = task_type
        self.rootAttribution = root_attribute
        # if self.run_id is None:
        #     run_obj = TestRun()
        #     self.run_id = run_obj.run.id
        #     self.compute_target = run_obj.compute
        #     self.region = run_obj.region
        #     self.experiment_id = run_obj.experiment.id
        #     self.subscription_id = run_obj.subscription
        #     self.parent_run_id = self.run_id
        #     if hasattr(run_obj.run, "parent"):
        #         self.parent_run_id = run_obj.run.parent.id
        #     self.rootAttribution = run_obj.root_attribute
        if self.task_type == "":
            import sys
            args = sys.argv
            if "--task" in args:
                ind = args.index("--task")
                self.task_type = sys.argv[ind+1]


class ModelImportHandler(logging.StreamHandler):
    """Model Import handler for stream handling."""

    def emit(self, record: logging.LogRecord) -> None:
        """Emits logs to stream after adding custom dimensions."""
        new_properties = getattr(record, "properties", {})
        new_properties.update({"log_id": str(uuid.uuid4())})
        custom_dimensions = CustomDimensions()
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


def get_logger(name, level=logging.DEBUG):
    logger = logging.getLogger(name)
    numeric_log_level = 1

    # don't log twice i.e. root logger
    logger.propagate = False
    logger.setLevel(numeric_log_level)
    handler_names = [handler.get_name() for handler in logger.handlers]
    app_name = Config.FINETUNE_APP_NAME

    if MODEL_IMPORT_HANDLER_NAME not in handler_names:
        ModelImportHandler()

    if APPINSIGHT_HANDLER_NAME not in handler_names:
        instrumentation_key = codecs.decode(INSTRUMENTATION_KEY, CODEC)
        appinsights_handler = get_telemetry_log_handler(
            instrumentation_key=instrumentation_key,
            component_name="",
        )

        formatter = AppInsightsPIIStrippingFormatter(
            fmt=(
                "%(asctime)s [{}] [{}] [%(module)s] %(funcName)s +%(lineno)s: %(levelname)-8s [%(process)d]"
                " %(message)s \n".format(app_name, run.id)
            )
        )

        appinsights_handler.setFormatter(formatter)
        appinsights_handler.setLevel(numeric_log_level)

        appinsights_handler.set_name("Testhandler")
        logger.addHandler(appinsights_handler)

    return logger


run = RunDetails()
