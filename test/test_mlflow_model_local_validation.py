# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for MLflow model local validation command execution."""

import importlib
import sys
import types
from pathlib import Path
from unittest.mock import Mock, patch


COMMON_SRC_DIR = Path(__file__).parents[1] / "assets" / "common" / "src"


def _install_import_stubs(monkeypatch):
    monkeypatch.syspath_prepend(str(COMMON_SRC_DIR))

    azure = types.ModuleType("azure")
    azure_ai = types.ModuleType("azure.ai")
    azure_ai_ml = types.ModuleType("azure.ai.ml")
    azure_ai_ml.MLClient = object

    azure_exceptions = types.ModuleType("azure.ai.ml.exceptions")
    azure_exceptions.ErrorTarget = types.SimpleNamespace()
    azure_exceptions.ErrorCategory = types.SimpleNamespace()
    azure_exceptions.MlException = Exception
    azure_exceptions.ValidationException = Exception

    azure_ml_identity = types.ModuleType("azure.ai.ml.identity")
    azure_ml_identity.AzureMLOnBehalfOfCredential = object

    azure_identity = types.ModuleType("azure.identity")
    azure_identity.ManagedIdentityCredential = object

    monkeypatch.setitem(sys.modules, "azure", azure)
    monkeypatch.setitem(sys.modules, "azure.ai", azure_ai)
    monkeypatch.setitem(sys.modules, "azure.ai.ml", azure_ai_ml)
    monkeypatch.setitem(sys.modules, "azure.ai.ml.exceptions", azure_exceptions)
    monkeypatch.setitem(sys.modules, "azure.ai.ml.identity", azure_ml_identity)
    monkeypatch.setitem(sys.modules, "azure.identity", azure_identity)

    azureml = types.ModuleType("azureml")
    azureml_common = types.ModuleType("azureml._common")
    azureml_error_definition = types.ModuleType("azureml._common._error_definition")
    azureml_error_definition.AzureMLError = object
    azureml_exceptions = types.ModuleType("azureml._common.exceptions")
    azureml_exceptions.AzureMLException = Exception
    azureml_core = types.ModuleType("azureml.core")
    azureml_core_run = types.ModuleType("azureml.core.run")
    azureml_core_run.Run = object

    monkeypatch.setitem(sys.modules, "azureml", azureml)
    monkeypatch.setitem(sys.modules, "azureml._common", azureml_common)
    monkeypatch.setitem(sys.modules, "azureml._common._error_definition", azureml_error_definition)
    monkeypatch.setitem(sys.modules, "azureml._common.exceptions", azureml_exceptions)
    monkeypatch.setitem(sys.modules, "azureml.core", azureml_core)
    monkeypatch.setitem(sys.modules, "azureml.core.run", azureml_core_run)

    config = types.ModuleType("utils.config")
    config.AppName = types.SimpleNamespace(MLFLOW_MODEL_LOCAL_VALIDATION="mlflow_model_local_validation")

    logging_utils = types.ModuleType("utils.logging_utils")
    logging_utils.custom_dimensions = types.SimpleNamespace(app_name=None)
    logging_utils.get_logger = lambda _: Mock()

    run_utils = types.ModuleType("utils.run_utils")
    run_utils.JobRunDetails = object

    exceptions = types.ModuleType("utils.exceptions")
    exceptions.ModelImportErrorStrings = types.SimpleNamespace()
    exceptions.UserIdentityMissingError = object
    exceptions.InvalidModelIDError = object
    exceptions.CondaEnvCreationError = object
    exceptions.CondaFileMissingError = object
    exceptions.MlflowModelValidationError = object
    exceptions.swallow_all_exceptions = lambda _: lambda func: func

    monkeypatch.setitem(sys.modules, "utils.config", config)
    monkeypatch.setitem(sys.modules, "utils.logging_utils", logging_utils)
    monkeypatch.setitem(sys.modules, "utils.run_utils", run_utils)
    monkeypatch.setitem(sys.modules, "utils.exceptions", exceptions)


def _reload_module(module_name: str):
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_validation_command_preserves_column_rename_map_as_single_argument(monkeypatch):
    """Verify shell metacharacters in column_rename_map are not interpreted by a shell."""
    _install_import_stubs(monkeypatch)
    validation = _reload_module("run_mlflow_model_local_validation")

    payload = "old:new'; touch /tmp/injected; echo 'still:data"
    cmd = validation._get_validation_command(
        model_dir=Path("/tmp/model"),
        test_data_path=Path("/tmp/data.csv"),
        col_rename_map_str=payload,
        task_name="summarization",
    )

    assert cmd[:6] == ["conda", "run", "-p", validation.ENV_PREFIX, "python", validation.SCRIPT_PATH]
    assert cmd[cmd.index("--column-rename-map") + 1] == payload
    assert validation.LOCAL_VALIDATION_OUT_FILE not in cmd
    assert "2>&1" not in cmd


def test_run_command_args_executes_without_shell(monkeypatch):
    """Verify argument-list commands are executed with shell disabled."""
    _install_import_stubs(monkeypatch)
    common_utils = _reload_module("utils.common_utils")

    completed_process = types.SimpleNamespace(returncode=0, stdout="ok")
    with patch.object(common_utils, "run", return_value=completed_process) as run_mock:
        exit_code, stdout = common_utils.run_command_args(["echo", "hello; touch /tmp/injected"])

    args, kwargs = run_mock.call_args
    assert args[0] == ["echo", "hello; touch /tmp/injected"]
    assert kwargs["shell"] is False
    assert exit_code == 0
    assert stdout == "ok"
