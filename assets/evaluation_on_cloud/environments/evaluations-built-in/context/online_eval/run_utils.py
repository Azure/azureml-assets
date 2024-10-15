# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AzureML Core Run utilities."""
from azureml.core import Run
from azureml.core.run import _OfflineRun

known_modules = {
    "online"
}


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
