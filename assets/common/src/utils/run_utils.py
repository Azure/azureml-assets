# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run Utils."""

from azureml.core import Run
from azureml.core.compute import ComputeTarget
from utils.config import LoggerConfig


class JobRunDetails:
    """Job Run details."""

    # static instance of RunDetails
    _instance = None

    def __init__(self):
        """Run details init. Should not be called directly and be instantiated via get_run_details."""
        self._run = Run.get_context()
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
        return self._run.id

    @property
    def parent_run_id(self):
        """Parent RunID of the existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._run.parent.id

    @property
    def details(self):
        """Run details of the existing run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        self._details = self._details or self._run.get_details()
        return self._details

    @property
    def workspace(self):
        """Return workspace."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self._run.experiment.workspace

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
        return self._run.experiment.id

    @property
    def subscription_id(self):
        """Return subcription ID."""
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
        """Return compute target for the current run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        return self.details.get("target", "")

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
            cpu_cluster = ComputeTarget(workspace=self._run.experiment.workspace, name=compute_name)
            return cpu_cluster.vm_size
        except Exception:
            return None

    @property
    def component_asset_id(self):
        """Run properties."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE
        run_properties = self.details.get("properties", {})
        return run_properties.get("azureml.moduleid", LoggerConfig.ASSET_NOT_FOUND)

    @property
    def input_assets(self):
        """Run properties."""
        if "OfflineRun" in self.run_id:
            return {}
        run_definition = self.details.get("runDefinition", {})
        return run_definition.get("inputAssets", {})

    @property
    def root_attribute(self):
        """Return root attribute of the run."""
        if "OfflineRun" in self.run_id:
            return LoggerConfig.OFFLINE_RUN_MESSAGE

        cur_attribute = self._run.id
        run = self._run.parent
        # update current run's root_attribute to the root run.
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
