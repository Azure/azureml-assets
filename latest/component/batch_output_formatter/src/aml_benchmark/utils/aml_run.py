# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AML run file."""
import os
import re
from typing import Dict

import mlflow
from azureml.core import Run
from azureml.core.run import _OfflineRun


class DummyWorkspace:
    """Dummy Workspace class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "local-ws"
        self.subscription_id = ""
        self.location = "local"
        self.resource_group = ""

    def get_mlflow_tracking_uri(self):
        """Mlflow Tracking URI.

        Returns:
            _type_: _description_
        """
        return mlflow.get_mlflow_tracking_uri()
        # get_tracking_uri()


class DummyExperiment:
    """Dummy Experiment class for offline logging."""

    def __init__(self):
        """__init__."""
        self.name = "offline_default_experiment"
        self.id = "1"
        self.workspace = DummyWorkspace()


class RunDetails:
    """Main class containing current Run's details."""

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
        """Azureml Run."""
        return self._run

    @property
    def experiment(self):
        """Azureml Experiment."""
        return self._experiment

    @property
    def workspace(self):
        """Azureml Workspace."""
        return self._workspace

    @property
    def compute(self):
        """Azureml compute instance."""
        if not isinstance(self._run, _OfflineRun):
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
        """Azure Region."""
        return self._workspace.location

    @property
    def subscription(self):
        """Azureml Subscription."""
        return self._workspace.subscription_id

    @property
    def parent_run(self):
        """Get Root run of the pipeline."""
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun) or (cur_run.parent is None):
            return self._run
        if cur_run.parent is not None:
            cur_run = cur_run.parent
        return cur_run

    @property
    def root_run(self):
        """Get Root run of the pipeline."""
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun) or (cur_run.parent is None):
            return self._run
        while cur_run.parent is not None:
            cur_run = cur_run.parent
        return cur_run

    @property
    def root_attribute(self):
        """Get Root attribute of the pipeline."""
        cur_run = self._run
        if isinstance(cur_run, _OfflineRun):
            return cur_run.id
        cur_attribute = self._run.id
        run = self._run.parent
        # update current run's root_attribute to the root run.
        while run is not None:
            cur_attribute = run.id
            run = run.parent
        return cur_attribute

    def get_extra_run_info(self) -> Dict[str, str]:
        """Get run details of the pipeline."""
        if isinstance(self._run, _OfflineRun):
            return {}
        raw_json = self._run.get_details()
        module_name = raw_json.get("properties", {}).get("azureml.moduleName", "")
        module_id = raw_json.get("properties", {}).get("azureml.moduleid", "")
        pipeline_type = raw_json.get("properties", {}).get("PipelineType", "")
        source = raw_json.get("properties", {}).get("runSource", "")
        try:
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT")
            location = re.compile("//(.*?)\\.").search(location).group(1)
        except Exception:
            location = os.environ.get("AZUREML_SERVICE_ENDPOINT", "")
        return {
            "moduleName": module_name,
            "moduleId": module_id,
            "pipeline_type": pipeline_type,
            "source": source,
            "location": location,
        }
