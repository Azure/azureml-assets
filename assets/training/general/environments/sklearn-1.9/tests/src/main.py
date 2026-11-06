# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate Scikit-learn 1.9 training in standard and Intel modes."""

import argparse
from pathlib import Path
import sys

import joblib
import mlflow
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split


def main(args):
    """Train, predict, and verify the saved model."""
    assert sys.version_info[:2] == (3, 11), sys.version
    assert sklearn.__version__.startswith("1.9."), sklearn.__version__
    if args.intel_extension:
        from sklearnex import patch_sklearn
        patch_sklearn()
    from sklearn.linear_model import LinearRegression

    mlflow.autolog()
    frame = pd.read_csv(args.diabetes_csv)
    x_train, x_test, y_train, _ = train_test_split(
        frame.drop(columns=["target"]), frame["target"], test_size=0.2, random_state=42
    )
    model = LinearRegression().fit(x_train, y_train)
    if args.intel_extension:
        assert type(model).__module__.startswith("sklearnex."), type(model)
    predictions = model.predict(x_test)
    assert predictions.shape == (len(x_test),)
    assert np.isfinite(predictions).all()
    outputs = Path("outputs")
    outputs.mkdir(exist_ok=True)
    model_path = outputs / f"model-intel-{args.intel_extension}.joblib"
    joblib.dump(model, model_path)
    np.testing.assert_allclose(joblib.load(model_path).predict(x_test), predictions)
    print(
        f"SKLEARN_1_9_TRAIN_RELOAD_PASSED intel={args.intel_extension} "
        f"python={sys.version.split()[0]} sklearn={sklearn.__version__}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diabetes-csv", required=True)
    parser.add_argument("--intel-extension", action="store_true")
    main(parser.parse_args())
