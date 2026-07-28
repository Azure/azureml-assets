# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for the online evaluation context."""
import importlib
import os
import sys

from promptflow.client import load_flow


def load_evaluator(evaluator):
    """Load the evaluator from the given path."""
    print(f"Loading evaluator {evaluator}")
    loaded_evaluator = load_flow(evaluator)
    print(loaded_evaluator)
    print(
        f"Loading module {os.getcwd()} {loaded_evaluator.entry.split(':')[0]} from {loaded_evaluator.path.parent.name}"
    )
    module_path = os.path.join(
        loaded_evaluator.path.parent, loaded_evaluator.entry.split(":")[0] + ".py"
    )
    module_name = loaded_evaluator.entry.split(":")[0]
    print(f"Loading module {module_name} from {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    print(f"Loaded module {mod}")
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    eval_class = getattr(mod, loaded_evaluator.entry.split(":")[1])
    return eval_class
