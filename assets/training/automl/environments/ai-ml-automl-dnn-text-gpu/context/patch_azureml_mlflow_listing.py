# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Backport MLflow 3 runs:/ model registration listing for azureml-mlflow."""

from pathlib import Path
import sysconfig


_ORIGINAL_LISTING = """\
        # get and filter by paths

        if path and self.artifacts.path and not path.startswith(self.artifacts.path):
"""
_PATCHED_LISTING = """\
        # get and filter by paths
        from mlflow.version import VERSION as MLFLOW_VERSION

        root_path = self.artifacts.path.rstrip("/") if self.artifacts.path else None
        if path is None and root_path and version.parse(MLFLOW_VERSION).major >= 3:
            # MLflow 3 lists the repository root while registering runs:/ models and expects
            # paths relative to that root, such as MLmodel.
            return [
                FileInfo(path=info.path[len(root_path) + 1:], is_dir=info.is_dir, file_size=info.file_size)
                for info in self.list_artifacts(root_path)
            ]

        if path and root_path and path != root_path and not path.startswith(self.artifacts.path):
"""


def patch_azureml_mlflow_listing(path):
    """Patch the installed azureml-mlflow, rejecting an unexpected upstream implementation."""
    source = path.read_text(encoding="utf-8")
    if source.count(_PATCHED_LISTING) == 1:
        return
    if source.count(_ORIGINAL_LISTING) != 1:
        raise RuntimeError("azureml-mlflow artifact listing changed; review the compatibility backport.")
    patched = source.replace(_ORIGINAL_LISTING, _PATCHED_LISTING)
    compile(patched, str(path), "exec")
    path.write_text(patched, encoding="utf-8")


if __name__ == "__main__":
    patch_azureml_mlflow_listing(
        Path(sysconfig.get_path("purelib"))
        / "azureml" / "mlflow" / "_store" / "artifact" / "artifact_repo.py"
    )
