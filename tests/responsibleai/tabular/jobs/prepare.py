# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import pandas as pd
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def fetch_and_write_boston_dataset():
    train_path = "./resources/boston_train/"
    test_path = "./resources/boston_test/"

    data = load_boston()
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
