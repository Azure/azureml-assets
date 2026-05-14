import re, os, ast

BASE = "/home/runner/work/azureml-assets/azureml-assets"

# Test the import scanner
def test_scanner():
    path = os.path.join(BASE, "assets/evaluators/builtin/f1_score/evaluator/_f1_score.py")
    with open(path) as f:
        content = f.read()
    lines = content.split('\n')
    for i, line in enumerate(lines):
        if (line.startswith('import ') or line.startswith('from ')) and not line.startswith('    '):
            print(f"  Top-level import at {i+1}: {line[:60]}")
        elif (line.strip().startswith('import ') or line.strip().startswith('from ')) and line.startswith(' '):
            print(f"  Indented import at {i+1}: {line[:60]}")

test_scanner()
