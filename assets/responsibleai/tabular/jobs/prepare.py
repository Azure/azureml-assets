# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resources')


def fetch_and_write_boston_dataset():
    train_path = os.path.join(data_dir, "boston_train")
    test_path = os.path.join(data_dir, "boston_test")

    data = fetch_openml(data_id=531)
    target_feature = "y"
    data_df = pd.DataFrame(data.data, columns=data.feature_names)

    X_train, X_test, y_train, y_test = train_test_split(
        data_df, data.target, test_size=0.2, random_state=7
    )

    train_data = X_train.copy()
    test_data = X_test.copy()
    train_data[target_feature] = y_train
    test_data[target_feature] = y_test

    print("Saving to files")
    train_data.to_parquet(os.path.join(train_path, "boston_train.parquet"), index=False)
    test_data.to_parquet(os.path.join(test_path, "boston_test.parquet"), index=False)

    return train_path, test_path


fetch_and_write_boston_dataset()
