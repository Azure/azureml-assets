# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from importlib.util import find_spec
from pathlib import Path

mlflow_sklearn_spec = find_spec("mlflow.sklearn")
mlflow_sklearn_path = Path(mlflow_sklearn_spec.origin)
source = mlflow_sklearn_path.read_text(encoding="utf-8")

serialization_default = "serialization_format=SERIALIZATION_FORMAT_SKOPS,"
if source.count(serialization_default) != 2:
    raise RuntimeError(
        "Unexpected MLflow sklearn serialization defaults"
    )
source = source.replace(
    serialization_default,
    "serialization_format=SERIALIZATION_FORMAT_CLOUDPICKLE,",
)

requirements_inference = (
    "            default_reqs = sorted("
    "set(inferred_reqs).union(default_reqs))\n"
)
if source.count(requirements_inference) != 1:
    raise RuntimeError(
        "Unexpected MLflow sklearn requirements inference"
    )
requirements_patch = (
    "            default_reqs.extend(\n"
    "                [\n"
    '                    f"mlflow-skinny=={mlflow.__version__}",\n'
    '                    _get_pinned_requirement("numpy"),\n'
    '                    _get_pinned_requirement("pandas"),\n'
    "                ]\n"
    "            )\n"
)
source = source.replace(
    requirements_inference,
    requirements_inference + requirements_patch,
)

mlflow_sklearn_path.write_text(source, encoding="utf-8")
