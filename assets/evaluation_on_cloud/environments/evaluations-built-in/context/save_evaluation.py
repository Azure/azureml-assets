# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Load a built-in or custom evulator as flow."""
import importlib
import logging
import os
import sys

from promptflow.client import load_flow

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def load_evaluator(evaluator):
    """Load evaluator as flow."""
    logger.info(f"Loading evaluator {evaluator}")
    loaded_evaluator = load_flow(evaluator)
    logger.info(f"Loaded evaluator {loaded_evaluator}")
    module_parent = loaded_evaluator.path.parent.name
    module_name = loaded_evaluator.entry.split(":")[0]
    module_path = os.path.join(os.getcwd(), module_parent, module_name + ".py")
    logger.info(f"Loading module {os.getcwd()} {module_name} from {module_parent}")
    logger.info(f"Loading module {module_name} from {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    logger.info(f"Loaded module {mod}")
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    eval_class = getattr(mod, loaded_evaluator.entry.split(":")[1])
    return eval_class
