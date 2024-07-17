# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate running RAIInsights using responsibleai docker image."""
# imports
import mlflow
import argparse

import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from responsibleai import RAIInsights, ModelTask

LABELS = 'labels'


def main(args):
    """Train model and then run RAIInsights."""
    # enable auto logging
    mlflow.autolog()

    # setup parameters
    params = {
        "fit_intercept": args.fit_intercept,
        "positive": args.positive,
    }

    # read in data
    df = pd.read_csv(args.diabetes_csv)

    # process data
    X_train, X_test, y_train, y_test = process_data(df, args.random_state)

    # train model
    model = train_model(params, X_train, X_test, y_train, y_test)

    # run RAIInsights
    X_train[LABELS] = y_train
    X_test[LABELS] = y_test
    task_type = ModelTask.REGRESSION
    rai_insights = RAIInsights(model, X_train, X_test, LABELS, task_type=task_type)
    rai_insights.error_analysis.add()
    rai_insights.counterfactual.add(total_CFs=20, desired_range=[50, 120])
    rai_insights.causal.add(treatment_features=['bmi', 'bp', 's2'])
    rai_insights.explainer.add()
    rai_insights.compute()
    print(rai_insights.error_analysis.get())
    print(rai_insights.counterfactual.get())
    print(rai_insights.causal.get())
    print(rai_insights.explainer.get())


def process_data(df, random_state):
    """Process data."""
    # split dataframe into X and y
    X = df.drop(["target"], axis=1)
    y = df["target"]

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # return splits and encoder
    return X_train, X_test, y_train, y_test


def train_model(params, X_train, X_test, y_train, y_test):
    """Create trained model."""
    # train model
    model = LinearRegression(**params)
    model = model.fit(X_train, y_train)

    # return model
    return model


def parse_args():
    """Parse arguments."""
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--diabetes-csv", type=str)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--fit_intercept", type=bool, default=True)
    parser.add_argument("--positive", type=bool, default=False)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
