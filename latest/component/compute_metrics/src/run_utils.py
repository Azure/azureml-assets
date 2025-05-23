# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AzureML Core Run utilities."""
from azureml.core import Run
from azureml.core.run import _OfflineRun
import azureml.evaluate.mlflow as aml_mlflow
import os
import re

known_modules = {
    'validation_trigger_model_evaluation',
    "model_prediction",
    "compute_metrics",
    "evaluate_model"
}


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
        return aml_mlflow.get_tracking_uri()


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
                info["moduleName"] = module_name if module_name in known_modules else 'Unknown'
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
