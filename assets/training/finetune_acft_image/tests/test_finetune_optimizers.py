# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import importlib.util
import sys
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


FINETUNE = Path(__file__).resolve().parents[1] / "src" / "finetune" / "finetune.py"
STUB_MODULES = (
    "deepspeed",
    "transformers.training_args",
    "transformers.trainer_utils",
    "azureml._common._error_definition.azureml_error",
    "azureml.acft.accelerator.constants",
    "azureml.acft.common_components.model_selector.constants",
    "azureml.acft.common_components.utils.error_handling.exceptions",
    "azureml.acft.common_components.utils.error_handling.error_definitions",
    "azureml.acft.common_components.utils.arg_utils",
    "azureml.acft.common_components.utils.error_handling.swallow_all_exceptions_decorator",
    "azureml.metrics.constants",
    "azureml.acft.image",
    "azureml.acft.image.components.common.constants",
    "azureml.acft.image.components.finetune.common.constants.constants",
    "azureml.acft.image.components.finetune.factory.mappings",
    "azureml.acft.image.components.finetune.factory.task_definitions",
    "azureml.acft.image.components.finetune.finetune_runner",
    "azureml.acft.image.components.finetune.defaults.training_defaults",
    "azureml.acft.common_components",
)


def load_finetune(monkeypatch, legacy):
    for name in STUB_MODULES:
        monkeypatch.setitem(sys.modules, name, MagicMock())
    monkeypatch.setitem(sys.modules, "optimum", None)
    names = (
        "ADAMW_TORCH", "ADAMW_TORCH_XLA", "ADAMW_BNB", "ADAMW_ANYPRECISION",
        "SGD", "ADAFACTOR", "ADAGRAD",
    )
    if legacy:
        names += ("ADAMW_HF", "ADAMW_APEX_FUSED")
    optimizers = Enum("OptimizerNames", {name: name.lower() for name in names})
    sys.modules["transformers.training_args"].OptimizerNames = optimizers
    sys.modules["transformers.trainer_utils"].IntervalStrategy = SimpleNamespace(
        NO="no", STEPS="steps", EPOCH="epoch"
    )
    sys.modules["azureml.acft.image.components.finetune.factory.task_definitions"].Tasks.MM_OBJECT_DETECTION = (
        "object_detection"
    )
    spec = importlib.util.spec_from_file_location("image_finetune_under_test", FINETUNE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, optimizers


@pytest.mark.parametrize("legacy", [False, True])
def test_optimizer_choices_follow_installed_transformers(monkeypatch, capsys, legacy):
    module, optimizers = load_finetune(monkeypatch, legacy)
    parser = module.get_parser()

    def parse(optimizer):
        return parser.parse_args([
            "--train_mltable_path", "train", "--task_name", "object_detection", "--optim", optimizer
        ])

    assert parse("adamw").optim == "adamw"
    assert module.Mapper.OPTIMIZER_MAP["adamw"] == optimizers.ADAMW_TORCH
    assert parse("adamw_ort_fused").optim == "adamw_ort_fused"
    if legacy:
        assert parse("adamw_hf").optim == "adamw_hf"
        assert module.Mapper.OPTIMIZER_MAP["adamw_hf"] == optimizers.ADAMW_HF
        assert module.Mapper.OPTIMIZER_MAP["adamw_apex_fused"] == optimizers.ADAMW_APEX_FUSED
    else:
        assert "adamw_hf" not in module.Mapper.OPTIMIZER_MAP
        assert "adamw_apex_fused" not in module.Mapper.OPTIMIZER_MAP
        with pytest.raises(SystemExit) as exc:
            parse("adamw_hf")
        assert exc.value.code == 2
        assert "invalid choice" in capsys.readouterr().err
