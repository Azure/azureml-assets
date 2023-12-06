import importlib
import os
import sys


def test_prs_code_importing():
    # Should point to the same root as the "code" in the batch-score component yamls
    component_code_root = "../../src"
    file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), component_code_root))
    sys.path.append(file_path)

    # Should be the exact same as the "entry_script" in the batch-score component yamls
    entry_script = "batch_score.main"

    module = importlib.import_module(entry_script, file_path)

    sys.path.pop()
    assert module

