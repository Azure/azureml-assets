# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate the supported skops subset against the AutoML NumPy stack."""

import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
import skops.io as skops_io
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def main():
    """Verify package versions and a representative safe model round trip."""
    if np.__version__ != "1.23.5":
        raise RuntimeError(f"Expected NumPy 1.23.5, found {np.__version__}.")
    skops_version = version("skops")
    if skops_version != "0.14.0":
        raise RuntimeError(f"Expected skops 0.14.0, found {skops_version}.")

    model = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("regressor", LinearRegression()),
        ]
    )
    features = np.array([[0.0], [1.0], [2.0], [3.0]])
    labels = np.array([1.0, 3.0, 5.0, 7.0])
    model.fit(features, labels)

    with tempfile.TemporaryDirectory() as temporary_directory:
        model_path = Path(temporary_directory) / "model.skops"
        skops_io.dump(model, model_path)
        untrusted_types = skops_io.get_untrusted_types(file=model_path)
        if untrusted_types:
            raise RuntimeError(
                f"Expected no untrusted types, found: {untrusted_types}."
            )
        loaded_model = skops_io.load(model_path, trusted=[])

    if not np.array_equal(model.predict(features), loaded_model.predict(features)):
        raise RuntimeError("The skops model round trip changed predictions.")


if __name__ == "__main__":
    main()
