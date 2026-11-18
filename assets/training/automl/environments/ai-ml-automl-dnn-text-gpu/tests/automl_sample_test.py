# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Smoke tests running a job in the ai-ml-automl-dnn-text-gpu environment."""
# Test NER runner imports, shared ONNX converters, and scikit-learn training.
# After the environment is deployed, notebooks tests should be run before updating
# the "Prod" label in the Jasmine service
import importlib
import os
import runpy
import sys
import time
import types
from pathlib import Path
from unittest.mock import Mock
import pytest
from azure.ai.ml import command, Input, MLClient
from azure.ai.ml.operations._run_history_constants import JobStatus
from azure.ai.ml.entities import Environment, BuildContext
from azure.identity import AzureCliCredential

BUILD_CONTEXT = Path("../context")
JOB_SOURCE_CODE = "src"
TIMEOUT_MINUTES = os.environ.get("timeout_minutes", 120)
STD_LOG = Path("artifacts/user_logs/std_log.txt")


def test_entrypoint_imports_are_deferred(tmp_path, monkeypatch):
    """Keep NLP logging importable without loading unrelated training dependencies."""
    package = tmp_path / "entrypoints_backport_test"
    utils = package / "utils"
    utils.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_text(
        "from .utils import common as entrypoint_util\n"
        "from .utils import training as training_entrypoint_util\n"
        "telemetry_preserved = True\n"
    )
    (utils / "__init__.py").write_text("")
    (utils / "common.py").write_text("def initialize_log_server():\n    return 'initialized'\n")
    (utils / "training.py").write_text("raise ImportError('incompatible tabular dependency')\n")
    backport = runpy.run_path(str(Path(__file__).parent / BUILD_CONTEXT / "patch_entrypoints.py"))
    backport["patch_entrypoints"](initializer)
    monkeypatch.syspath_prepend(str(tmp_path))

    entrypoints = importlib.import_module(package.name)
    common = importlib.import_module(package.name + ".utils.common")
    assert common.initialize_log_server() == "initialized"
    assert entrypoints.entrypoint_util is common
    assert entrypoints.__dict__["entrypoint_util"] is common
    assert entrypoints.telemetry_preserved
    with pytest.raises(AttributeError):
        entrypoints.unknown_entrypoint
    with pytest.raises(ImportError, match="incompatible tabular dependency"):
        entrypoints.training_entrypoint_util


def test_entrypoint_backport_rejects_changed_imports(tmp_path):
    """Fail the build instead of silently patching an unexpected SDK initializer."""
    initializer = tmp_path / "__init__.py"
    source = "from .utils import changed_entrypoint\n"
    initializer.write_text(source)
    backport = runpy.run_path(str(Path(__file__).parent / BUILD_CONTEXT / "patch_entrypoints.py"))
    with pytest.raises(RuntimeError, match="Unexpected AutoML entrypoint import"):
        backport["patch_entrypoints"](initializer)
    assert initializer.read_text() == source


@pytest.mark.parametrize(
    ("mlflow_version", "use_legacy"),
    [("1.30.0", False), ("2.22.4", False), ("3.0.0", True), ("3.15.0", True), ("3.18.0", True)],
)
def test_mlflow_loader_backport(tmp_path, monkeypatch, mlflow_version, use_legacy):
    """Keep loader logging compatible with both MLflow 2 and MLflow 3."""
    source = (
        "def log_model(self, mlflow, artifact_path, loader_module, data_path, signature, "
        "input_example, conda_env, metadata):\n"
        "    if True:\n"
        "        if True:\n"
        "            mlflow.pyfunc.log_model(\n"
        "                artifact_path=artifact_path,\n"
        "                loader_module=loader_module,\n"
        "                data_path=data_path,\n"
        "                signature=signature,\n"
        "                input_example=input_example,\n"
        "                conda_env=conda_env,\n"
        "                metadata=metadata)\n"
    )
    module_path = tmp_path / "automl_run_context.py"
    module_path.write_text(source)
    backport = runpy.run_path(str(Path(__file__).parent / BUILD_CONTEXT / "patch_mlflow_logging.py"))
    backport["patch_mlflow_logging"](module_path)
    patched = module_path.read_text()
    backport["patch_mlflow_logging"](module_path)
    assert module_path.read_text() == patched

    mlflow = types.ModuleType("mlflow")
    mlflow.__version__ = mlflow_version
    mlflow.pyfunc = types.SimpleNamespace(log_model=Mock())
    models = types.ModuleType("mlflow.models")
    models.Model = types.SimpleNamespace(_log_v2=Mock())
    monkeypatch.setitem(sys.modules, "mlflow", mlflow)
    monkeypatch.setitem(sys.modules, "mlflow.models", models)
    namespace = {}
    exec(compile(patched, str(module_path), "exec"), namespace)

    options = {
        "artifact_path": "outputs/mlflow-model",
        "loader_module": "ner_loader",
        "data_path": "model.pkl",
        "signature": object(),
        "input_example": object(),
        "conda_env": {"name": "ner"},
        "metadata": {"task": "ner"},
    }
    namespace["log_model"](types.SimpleNamespace(run_id="parent_1"), mlflow, **options)

    if use_legacy:
        models.Model._log_v2.assert_called_once_with(flavor=mlflow.pyfunc, run_id="parent_1", **options)
        mlflow.pyfunc.log_model.assert_not_called()
    else:
        mlflow.pyfunc.log_model.assert_called_once_with(**options)
        models.Model._log_v2.assert_not_called()


def test_mlflow_loader_backport_rejects_changed_source(tmp_path):
    """Reject an upstream runtime whose loader call has changed."""
    module_path = tmp_path / "automl_run_context.py"
    source = "def changed_runtime():\n    pass\n"
    module_path.write_text(source)
    backport = runpy.run_path(str(Path(__file__).parent / BUILD_CONTEXT / "patch_mlflow_logging.py"))
    with pytest.raises(RuntimeError, match="AutoML MLflow loader logging changed"):
        backport["patch_mlflow_logging"](module_path)
    assert module_path.read_text() == source


def test_azure_ai_ml_automl():
    """Tests a sample job using ai-ml-automl-dnn-text-gpu as the environment."""
    this_dir = Path(__file__).parent

    subscription_id = os.environ.get("subscription_id")
    resource_group = os.environ.get("resource_group")
    workspace_name = os.environ.get("workspace")

    ml_client = MLClient(
        AzureCliCredential(), subscription_id, resource_group, workspace_name
    )

    env_name = "ai-ml-automl-dnn-text-gpu"

    env_docker_context = Environment(
        build=BuildContext(path=this_dir / BUILD_CONTEXT),
        name="ai-ml-automl-dnn-text-gpu",
        description="ai-ml-automl-dnn-text-gpu environment created from a Docker context.",
    )
    ml_client.environments.create_or_update(env_docker_context)

    # create the command
    job = command(
        code=this_dir / JOB_SOURCE_CODE,  # local path where the code is stored
        command="python main.py --diabetes-csv ${{inputs.diabetes}}",
        inputs={
            "diabetes": Input(
                type="uri_file",
                path="https://azuremlexamples.blob.core.windows.net/datasets/diabetes.csv",
            )
        },
        environment=f"{env_name}@latest",
        compute=os.environ.get("gpu_cluster"),
        display_name="sklearn-diabetes-example",
        description="A test run of the ai-ml-automl-dnn-text-gpu curated environment",
        experiment_name="sklearnExperiment",
    )

    returned_job = ml_client.create_or_update(job)
    assert returned_job is not None

    # Poll until final status is reached or timed out
    timeout = time.time() + (TIMEOUT_MINUTES * 60)
    while time.time() <= timeout:
        job = ml_client.jobs.get(returned_job.name)
        status = job.status
        if status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            break
        time.sleep(30)  # sleep 30 seconds
    else:
        # Timeout
        ml_client.jobs.begin_cancel(returned_job.name)
        raise Exception(
            f"Test aborted because the job took longer than {TIMEOUT_MINUTES} minutes. "
            f"Last status was {status}."
        )

    if status == JobStatus.FAILED:
        ml_client.jobs.download(returned_job.name)
        if STD_LOG.exists():
            print(f"*** BEGIN {STD_LOG} ***")
            with open(STD_LOG, "r") as f:
                print(f.read(), end="")
            print(f"*** END {STD_LOG} ***")
        else:
            ml_client.jobs.stream(returned_job.name)

    assert status == JobStatus.COMPLETED
