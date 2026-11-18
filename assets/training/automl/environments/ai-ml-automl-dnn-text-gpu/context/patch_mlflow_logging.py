# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Backport legacy MLflow logging for AutoML custom-loader models."""

from pathlib import Path
import sysconfig


_ORIGINAL_CALL = "            mlflow.pyfunc.log_model(\n"
_PATCHED_CALL = """\
            from packaging.version import Version

            if Version(mlflow.__version__).major >= 3:
                from functools import partial
                from mlflow.models import Model

                # Azure ML's MLflow service only supports legacy model logging.
                log_model = partial(Model._log_v2, flavor=mlflow.pyfunc, run_id=self.run_id)
            else:
                log_model = mlflow.pyfunc.log_model
            log_model(
"""


def patch_mlflow_logging(path):
    """Patch the installed runtime, rejecting an unexpected upstream implementation."""
    source = path.read_text(encoding="utf-8")
    if source.count(_PATCHED_CALL) == 1:
        return
    if source.count(_ORIGINAL_CALL) != 1:
        raise RuntimeError("AutoML MLflow loader logging changed; review the compatibility backport.")
    patched = source.replace(_ORIGINAL_CALL, _PATCHED_CALL)
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    patch_mlflow_logging(
        Path(sysconfig.get_path("purelib"))
        / "azureml" / "automl" / "runtime" / "automl_run_context.py"
    )
