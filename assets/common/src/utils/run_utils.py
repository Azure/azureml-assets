# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Utils."""

import json
from utils.config import LoggerConfig
import os
from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ManagedIdentityCredential, DefaultAzureCredential
from azure.ai.ml.exceptions import ErrorTarget, ErrorCategory, MlException
from utils.exceptions import ModelImportErrorStrings

def get_mlclient():
    """Return ML Client."""
    has_msi_succeeded = False
    try:
        msi_client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
        credential = ManagedIdentityCredential(client_id=msi_client_id)
        credential.get_token("https://management.azure.com/.default")
        has_msi_succeeded = True
    except Exception:
        # Fall back to AzureMLOnBehalfOfCredential in case ManagedIdentityCredential does not work
        has_msi_succeeded = False
    if not has_msi_succeeded:
        try:
            credential = AzureMLOnBehalfOfCredential()
            # Check if given credential can get token successfully.
            credential.get_token("https://management.azure.com/.default")
        except Exception as ex:
            message = ModelImportErrorStrings.USER_IDENTITY_MISSING_ERROR
            raise MlException(
                message=message.format(ex=ex), no_personal_data_message=message,
                error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.IDENTITY,
                error=ex
            )
    try:
        subscription_id = os.environ.get("AZUREML_ARM_SUBSCRIPTION")
        resource_group = os.environ.get("AZUREML_ARM_RESOURCEGROUP")
        workspace = os.environ.get("AZUREML_ARM_WORKSPACE_NAME")
    except Exception as ex:
        message = "Failed to get AzureML ARM env variable : {ex}"
        raise MlException(
            message=message.format(ex=ex), no_personal_data_message=message,
            error_category=ErrorCategory.SYSTEM_ERROR, target=ErrorTarget.COMPONENT,
            error=ex
        )
    return MLClient(
        credential= credential,
        subscription_id=subscription_id,
        resource_group_name=resource_group,
        workspace_name=workspace,
    )

class JobRunDetails:
    """Job Run details."""

    # static instance of RunDetails
    _instance = None

    def __init__(self):
        """Run details init. Should not be called directly and be instantiated via get_run_details."""
        self._ml_client = get_mlclient()
        self._job = self._ml_client.jobs.get(os.environ["AZUREML_RUN_ID"])
        self._details = None

    @staticmethod
    def get_run_details():
        """Get JobRunDetails details. This should be called instead of calling JobRunDetails constructor."""
        if not JobRunDetails._instance:
            JobRunDetails._instance = JobRunDetails()
        return JobRunDetails._instance

    @property
    def run_id(self):
        """Run ID of the existing run."""
        return self._job.name

    @property
    def parent_run_id(self):
        """Parent RunID of the existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        #return self._job.parent if self._job.parent else "No parent job id"
        return getattr(self._job, "parent", None) or "No parent job id"

    @property
    def details(self):
        """Run details of the existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        self._details = self._details or self._ml_client.jobs.get(self.run_id)
        return self._details

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
        #run_properties = self.details.get("properties", {})
        run_properties = self.details._to_dict().get("properties", {})
        return run_properties.get("azureml.moduleid", LoggerConfig.ASSET_NOT_FOUND)
        

    @property
    def root_attribute(self):
        """Return root attribute of the run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE

        cur_attribute = self._job.id
        # run = self._job.parent
        run = getattr(self._job, "parent", None)
        #update current run's root_attribute to the root run.
        while run is not None:
            cur_attribute = run.id
            run = run.parent
        return cur_attribute

    def __str__(self):
        """Job Run details to string."""
        return (
            "JobRunDetails:\n"
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
