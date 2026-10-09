"""Apply compatibility fixes to responsibleai-tabular-automl."""

from importlib.util import find_spec
from pathlib import Path


package_spec = find_spec("responsibleai_tabular_automl")
if package_spec is None or package_spec.submodule_search_locations is None:
    raise RuntimeError("responsibleai-tabular-automl is not installed")

module_path = (
    Path(next(iter(package_spec.submodule_search_locations))) / "rai_automl.py"
)
content = module_path.read_text()

replacements = {
    '''    train_predictions = pd.read_parquet(
        "outputs/rai/predictions.npy.parquet"
    ).values
''': '''    train_predictions = pd.read_parquet(
        "outputs/rai/predictions.npy.parquet"
    ).values.ravel()
''',
    '''    test_predictions = pd.read_parquet(
        "outputs/rai/predictions_test.npy.parquet"
    ).values
''': '''    test_predictions = pd.read_parquet(
        "outputs/rai/predictions_test.npy.parquet"
    ).values.ravel()
''',
}

for original, replacement in replacements.items():
    if replacement in content:
        continue
    if content.count(original) != 1:
        raise RuntimeError(
            f"Expected one matching prediction read in {module_path}"
        )
    content = content.replace(original, replacement)

module_path.write_text(content)
