# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.


import os
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')


def fetch_and_write_iris_dataset():
    baseline_path = os.path.join(data_dir, "iris_baseline")
    production_path = os.path.join(data_dir, "iris_production")

    data = load_iris()
    target_feature = "target"
    data_df = pd.DataFrame(data.data, columns=data.feature_names)

    # use train_test_split to split dataset into baseline and production
    X_baseline, X_production, y_baseline, y_production = train_test_split(
        data_df, data.target, test_size=0.5, random_state=7
    )

    baseline_data = X_baseline.copy()
    production_data = X_production.copy()
    baseline_data[target_feature] = y_baseline
    production_data[target_feature] = y_production

    print("Saving to files")
    baseline_data.to_parquet(os.path.join(baseline_data, "iris_baseline.parquet"), index=False)
    production_data.to_parquet(os.path.join(production_data, "iris_production.parquet"), index=False)

    return baseline_path, production_path


fetch_and_write_iris_dataset()
