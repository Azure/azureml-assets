from azureml.core import Run
from azureml.core.run import _OfflineRun
import mlflow

class DummyWorkspace:
    def __init__(self):
        self.name = "local-ws"
        self.subscription_id = ""
        self.location = "local"
    
    def get_mlflow_tracking_uri(self):
        return mlflow.get_tracking_uri()

class DummyExperiment:
    def __init__(self):
        self.name = "offline_default_experiment"
        self.id = "1"
        self.workspace = DummyWorkspace()

class TestRun:
    def __init__(self):
        self._run = Run.get_context()
        if isinstance(self._run, _OfflineRun):
            self._experiment = DummyExperiment()
            self._workspace = self._experiment.workspace
        else:
            self._experiment = self._run.experiment
            self._workspace = self._experiment.workspace
    
    @property
    def run(self):
        return self._run
    
    @property
    def experiment(self):
        return self._experiment
    
    @property
    def workspace(self):
        return self._workspace
    
    @property
    def compute(self):
        if not isinstance(self._run, _OfflineRun):
            return self._run.get_details()["target"]
        return "local"
    
    @property
    def region(self):
        return self._workspace.location
    
    @property
    def subscription(self):
        return self._workspace.subscription_id