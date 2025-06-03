import sys
from pathlib import Path
import os
import importlib

def test_evaluate_obo_sets_env_var():
    script_path = Path(__file__).parent.parent / "context" / "evaluate_on_data.py"
    module_name = "evaluate_on_data"
    sys.path.insert(0, str(script_path.parent))

    if module_name in sys.modules:
        del sys.modules[module_name]

    assert os.environ.get("AZUREML_OBO_ENABLED") is None

    importlib.import_module(module_name)

    assert os.environ.get("AZUREML_OBO_ENABLED") == "True"
