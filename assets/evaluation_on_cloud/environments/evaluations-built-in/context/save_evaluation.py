"""Load a built-in or custom evulator as flow."""
import importlib
import logging
import os
import sys
from promptflow.client import load_flow

logger = logging.getLogger(__name__)


def load_evaluator(evaluator):
    """Load evaluator as flow."""
    logger.info(f"Loading evaluator {evaluator}")
    loaded_evaluator = load_flow(evaluator)
    logger.info(loaded_evaluator)
    logger.info(f"Loading module {os.getcwd()} {loaded_evaluator.entry.split(':')[0]} from {loaded_evaluator.path.parent.name}")
    module_path = os.path.join(os.getcwd(), loaded_evaluator.path.parent.name,
                               loaded_evaluator.entry.split(":")[0] + ".py")
    module_name = loaded_evaluator.entry.split(":")[0]
    logger.info(f"Loading module {module_name} from {module_path}")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    mod = importlib.util.module_from_spec(spec)
    logger.info(f"Loaded module {mod}")
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    eval_class = getattr(mod, loaded_evaluator.entry.split(":")[1])
    return eval_class
