# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Logging utils."""

from azureml.model.mgmt.config import AppName, LoggerConfig
from azure.ai.ml import MLClient
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException
from azureml.model.mgmt.utils.exceptions import ModelImportErrorStrings
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential
import platform
import uuid
import logging
import sys
import os


def get_mlclient():
    """Return ML Client."""
    has_msi_succeeded = False
    logger = get_logger(__name__)
    try:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
        credential.get_token("https://management.azure.com/.default")
        has_msi_succeeded = True
    except Exception:
        # Fall back to AzureMLOnBehalfOfCredential in case ManagedIdentityCredential does not work
        has_msi_succeeded = False
        logger.warning("ManagedIdentityCredential was not found in the compute. "
                       "Falling back to AzureMLOnBehalfOfCredential")

    if not has_msi_succeeded:
        try:
            credential = AzureMLOnBehalfOfCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            message = ModelImportErrorStrings.USER_IDENTITY_MISSING_ERROR
            raise MlException(
                message=message, no_personal_data_message=message,
                error_category=ErrorCategory.USER_ERROR, target=ErrorTarget.COMPONENT,
                error=ex
            )
    try:
        subscription_id = os.environ['AZUREML_ARM_SUBSCRIPTION']
        resource_group = os.environ["AZUREML_ARM_RESOURCEGROUP"]
        workspace = os.environ["AZUREML_ARM_WORKSPACE_NAME"]
    except Exception as ex:
        message = "Failed to get AzureML ARM env variable : {ex}"
        raise MlException(
            message=message.format(ex=ex), no_personal_data_message=message,
            error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.COMPONENT,
            error=ex
        )

    return MLClient(
        credential=credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

class RunDetails:
    """RunDetails."""

    def __init__(self):
        """Run details init."""
        self._run_details = None
        self._ml_client = get_mlclient()
        self._job = self._ml_client.jobs.get(os.environ["AZUREML_RUN_ID"])
    @property
    def run_id(self):
        """Run ID of the existing run."""
        return self._job.name

    @property
    def parent_run_id(self):
        """Parent RunID of the existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._job.parent if self._job.parent else "No parent job id"

    @property
    def run_details(self):
        """Run details of the existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        self._details = self._details or self._ml_client.jobs.get(self.run_id)
        return self._run_details

    @property
    def workspace(self):
        """Return workspace."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._ml_client.workspaces.get(name=self._ml_client._operation_scope.workspace_name)

    @property
    def workspace_name(self):
        """Return workspace name."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self.workspace.name

    @property
    def experiment_id(self):
        """Return experiment id."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._job.experiment_name

    @property
    def subscription_id(self):
        """Return subcription ID."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._ml_client._operation_scope.subscription_id

    @property
    def region(self):
        """Return the region where the run is executing."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self.workspace.location

    @property
    def compute(self):
        """Return compute target for the current run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._job.compute if self._job.compute else ""

    @property
    def vm_size(self):
        """Return compute VM size."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        compute_name = self.compute
        if compute_name == "":
            return "No compute found."

        # TODO: Use V2 way of determining this.
        try:
            cpu_cluster = self._ml_client.compute.get(compute_name)
            return cpu_cluster.properties.vm_size
        except Exception:
            return None

    @property
    def component_asset_id(self):
        """Run properties."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        run_properties = self.run_details.get("properties", {})
        return run_properties.get("azureml.moduleid", LoggerConfig.ASSET_NOT_FOUND)

    @property
    def root_attribute(self):
        """Return root attribute of the run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE

        cur_attribute = self._job.id
        run = self._job.parent
        # update current run's root_attribute to the root run.
        while run is not None:
            cur_attribute = run.id
            run = run.parent
        return cur_attribute

    def __str__(self):
        """Run details to string."""
        return (
            "RunDetails:\n"
            + f"\nrun_id: {self.run_id},\n"
            + f"parent_run_id: {self.parent_run_id},\n"
            + f"subscription_id: {self.subscription_id},\n"
            + f"workspace_name: {self.workspace_name},\n"
            + f"root_attribute: {self.root_attribute},\n"
            + f"experiment_id: {self.experiment_id},\n"
            + f"region: {self.region},\n"
            + f"compute: {self.compute},\n"
            + f"vm_size: {self.vm_size},\n"
            + f"component_asset_id : {self.component_asset_id}\n"
        )


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

    return logger


run_details = RunDetails()
custom_dimensions = CustomDimensions(run_details=run_details)
