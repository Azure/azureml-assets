# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Patch AutoML runtime sparse-data helpers for pandas 2 compatibility.

The AutoML 1.62.0 runtime still references the pandas 1.x
``pd.SparseDataFrame`` and ``pd.SparseSeries`` attributes. The ai-ml-automl
image intentionally force-installs pandas 2.0.0 for security reasons, where
those attributes no longer exist. Patch only the affected runtime files in
place until the runtime wheels contain the fix.
"""

import py_compile
import site
from pathlib import Path


def _site_packages() -> Path:
    for path in site.getsitepackages():
        candidate = Path(path)
        if (candidate / "azureml").exists():
            return candidate
    raise RuntimeError("Unable to locate AzureML site-packages directory.")


def _replace_once(path: Path, old: str, new: str, fixed_markers: tuple[str, ...]) -> None:
    text = path.read_text(encoding="utf-8")
    if new in text or all(marker in text for marker in fixed_markers):
        print(f"{path}: already patched")
        return
    if old not in text:
        raise RuntimeError(
            f"{path}: expected patch target was not found and fixed markers "
            "were not all present. The package source likely changed; review "
            "whether this pandas 2 patch is still needed."
        )
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    print(f"{path}: patched")


def _patch_runtime_utilities(site_packages: Path) -> None:
    path = site_packages / "azureml/automl/runtime/shared/utilities.py"
    old = """def issparse(obj: Any) -> bool:
    \"\"\"
    Check whether an object is sparse or not.

    :param obj: The input object.
    :return: Whether an object is sparse or not.
    \"\"\"
    if obj is None:
        return False

    return isinstance(obj, pd.SparseDataFrame) or scipy.sparse.issparse(obj)
"""
    new = """def _is_pandas_sparse(obj: Any) -> bool:
    legacy_sparse_dataframe = getattr(pd, \"SparseDataFrame\", None)
    legacy_sparse_series = getattr(pd, \"SparseSeries\", None)

    if legacy_sparse_dataframe is not None and isinstance(obj, legacy_sparse_dataframe):
        return True
    if legacy_sparse_series is not None and isinstance(obj, legacy_sparse_series):
        return True
    if isinstance(obj, pd.Series):
        return isinstance(obj.dtype, pd.SparseDtype)
    if isinstance(obj, pd.DataFrame):
        return len(obj.dtypes) > 0 and all(isinstance(dtype, pd.SparseDtype) for dtype in obj.dtypes)

    return False


def issparse(obj: Any) -> bool:
    \"\"\"
    Check whether an object is sparse or not.

    :param obj: The input object.
    :return: Whether an object is sparse or not.
    \"\"\"
    if obj is None:
        return False

    return scipy.sparse.issparse(obj) or _is_pandas_sparse(obj)
"""
    _replace_once(
        path,
        old,
        new,
        fixed_markers=("_is_pandas_sparse", "pd.SparseDtype", "scipy.sparse.issparse(obj)"),
    )


def _patch_lagging_transformer(site_packages: Path) -> None:
    path = site_packages / "azureml/training/tabular/featurization/timeseries/lagging_transformer.py"
    old = """        raw_feature_names = None
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.SparseDataFrame):
            raw_feature_names = x.columns
"""
    new = """        raw_feature_names = None
        legacy_sparse_dataframe = getattr(pd, \"SparseDataFrame\", None)
        if isinstance(x, pd.DataFrame) or (
            legacy_sparse_dataframe is not None and isinstance(x, legacy_sparse_dataframe)
        ):
            raw_feature_names = x.columns
"""
    _replace_once(
        path,
        old,
        new,
        fixed_markers=("legacy_sparse_dataframe", "getattr(pd, \"SparseDataFrame\", None)"),
    )


def _patch_featurization_phase(site_packages: Path) -> None:
    path = site_packages / "azureml/train/automl/runtime/_automl_job_phases/featurization_phase.py"
    text = path.read_text(encoding="utf-8")
    if "from azureml.automl.runtime.shared.utilities import issparse" not in text:
        raise RuntimeError(f"{path}: expected AutoML runtime issparse import was not found")

    old_helper_anchor = """logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class FeaturizationPhase:
"""
    new_helper_anchor = """logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _to_dense_if_sparse(data):
    if data is not None and issparse(data):
        if hasattr(data, \"sparse\") and hasattr(data.sparse, \"to_dense\"):
            return data.sparse.to_dense()
        return data.todense()

    return data


class FeaturizationPhase:
"""
    _replace_once(
        path,
        old_helper_anchor,
        new_helper_anchor,
        fixed_markers=("_to_dense_if_sparse", "data.sparse.to_dense()"),
    )

    old_conversion = (
        "                if automl_settings.iterations == "
        "constants.RuleBasedValidation.AUTOFEATURIZATION_ITERATION_COUNT:\n"
        "                    if issparse(td_ctx.X):\n"
        "                        td_ctx.X = td_ctx.X.todense()\n"
        "                    if issparse(td_ctx.X_valid):\n"
        "                        td_ctx.X_valid = td_ctx.X_valid.todense()\n"
    )
    new_conversion = (
        "                if automl_settings.iterations == "
        "constants.RuleBasedValidation.AUTOFEATURIZATION_ITERATION_COUNT:\n"
        "                    td_ctx.X = _to_dense_if_sparse(td_ctx.X)\n"
        "                    td_ctx.X_valid = _to_dense_if_sparse(td_ctx.X_valid)\n"
    )
    _replace_once(
        path,
        old_conversion,
        new_conversion,
        fixed_markers=(
            "td_ctx.X = _to_dense_if_sparse(td_ctx.X)",
            "td_ctx.X_valid = _to_dense_if_sparse(td_ctx.X_valid)",
        ),
    )


def main() -> None:
    """Apply the AutoML pandas 2 sparse compatibility patch."""
    site_packages = _site_packages()
    print(f"Patching AutoML runtime under {site_packages}")
    _patch_runtime_utilities(site_packages)
    _patch_lagging_transformer(site_packages)
    _patch_featurization_phase(site_packages)
    print("AutoML pandas 2 sparse compatibility patch complete.")


if __name__ == "__main__":
    main()
