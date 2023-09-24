# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Data Drift Compute Histogram Component."""

import argparse
import pandas as pd
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    FloatType
)
from shared_utilities.io_utils import try_read_mltable_in_spark_with_warning, save_spark_df_as_mltable, init_spark
from shared_utilities import constants

from responsibleai import RAIInsights, FeatureMetadata
from ml_wrappers.model.predictions_wrapper import (
    PredictionsModelWrapperClassification,
    PredictionsModelWrapperRegression)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    pass

from feature_importance_metrics.feature_importance_utilities import (
    compute_categorical_features, convert_pandas_to_spark, log_time_and_message)


def parse_args():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_data", type=str)
    parser.add_argument("--target_column", type=str, required=False)
    parser.add_argument("--task_type", type=str, required=False)
    parser.add_argument("--signal_metrics", type=str)

    args = parser.parse_args()

    return args


def determine_task_type(target_column, baseline_data):
    """Determine the task type based on the type of the target column.

    :param target_column: the column to predict
    :type target_column: string
    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :return: task type, either regression or classification
    :rtype: string
    """
    baseline_column = pd.Series(baseline_data[target_column])
    baseline_column_type = baseline_column.dtype.name
    if baseline_column_type == "float64":
        return constants.REGRESSION
    if baseline_column_type == "object" or baseline_column_type == "bool":
        return constants.CLASSIFICATION
    if baseline_column_type == "int64":
        distinct_column_values = len(baseline_column.unique())
        total_column_values = len(baseline_column)
        distinct_value_ratio = distinct_column_values / total_column_values
        if distinct_value_ratio < 0.05:
            return constants.CLASSIFICATION
        else:
            return constants.REGRESSION


def create_lightgbm_model(X, y, task_type):
    """Create model on which to calculate feature importances.

    :param X: x values (data excluding target columns)
    :type X: pandas.Dataframe
    :param y: nparray of y values (target column data)
    :type y: nparray
    :param task_type: the task type (regression or classification) of the resulting model
    :type task_type: string
    :return: an appropriate model wrapper
    :rtype: LightGBMClassifier or LightGBMRegressor
    """
    if task_type == constants.CLASSIFICATION:
        lgbm = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1,
                              max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    else:
        lgbm = LGBMRegressor(boosting_type='gbdt', learning_rate=0.1,
                             max_depth=5, n_estimators=200, n_jobs=1, random_state=777)

    model = lgbm.fit(X, y)
    log_time_and_message(f"Created lightgbm model using task_type: {task_type}")
    return model


def get_model_wrapper(task_type, target_column, baseline_data):
    """Create model wrapper using ml-wrappers on which to calculate feature importances.

    :param task_type: The task type (regression or classification) of the resulting model
    :type task_type: string
    :param target_column: the column to predict
    :type target_column: string
    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :return: an appropriate model wrapper
    :rtype: PredictionsModelWrapperRegression or PredictionsModelWrapperClassification
    """
    y_train = baseline_data[target_column]
    x_train = baseline_data.drop([target_column], axis=1)
    # Transform categorical features into the appropriate type that is expected by LightGBM
    for column in x_train:
        col_type = x_train[column].dtype.name
        if col_type == 'object' or col_type == 'category':
            x_train[column] = x_train[column].astype('category')
    model = create_lightgbm_model(x_train, y_train, task_type)
    model_predict = model.predict(x_train)

    if task_type == constants.CLASSIFICATION:
        model_predict_proba = model.predict_proba(x_train)
        model_wrapper = PredictionsModelWrapperClassification(
            x_train,
            model_predict,
            model_predict_proba)
    else:
        model_wrapper = PredictionsModelWrapperRegression(x_train, model_predict)

    log_time_and_message("Created ml wrapper")
    return model_wrapper


def compute_explanations(model_wrapper, data, categorical_features, target_column, task_type):
    """Compute explanations (feature importances) for a given dataset.

    :param model_wrapper: wrapper around a model that can be used to calculate explanations
    :type model_wrapper: PredictionsModelWrapperRegression or PredictionsModelWrapperClassification
    :param  data: The data used to calculate the explanations
    :type data: pandas.Dataframe
    :param categorical_features: categorical features not including the target column
    :type categorical_features: list[str]
    :param target_column: the column to predict
    :type target_column: string
    :param task_type: the task type (regression or classification) of the resulting model
    :type task_type: string
    :return: explanation scores for the input data
    :rtype: list[float]
    """
    # Create the RAI Insights object, use baseline as train and test data
    feature_metadata = FeatureMetadata(categorical_features=categorical_features, dropped_features=[])
    rai_i: RAIInsights = RAIInsights(
        model_wrapper, data, data, target_column, task_type, feature_metadata=feature_metadata
    )
    log_time_and_message("Created RAIInsights")
    # Add the global explanations using batching to allow for larger input data sizes
    rai_i.explainer.add()
    evaluation_data = data.drop([target_column], axis=1)
    log_time_and_message("Requesting explanations")
    explanationData = rai_i.explainer.request_explanations(local=False, data=evaluation_data)
    return explanationData.precomputedExplanations.globalFeatureImportance['scores']


def compute_feature_importance(task_type, target_column, baseline_data, categorical_features):
    """Compute feature importance of baseline data.

    :param task_type: The task type (regression or classification) of the resulting model
    :type task_type: string
    :param target_column: the column to predict
    :type target_column: string
    :param baseline_data: The baseline data meaning the data used to create the
    model monitor
    :type baseline_data: pandas.DataFrame
    :param categorical_features: The column names which are categorical in type
    :type categorical_features: list[string]
    :return: list of feature importances in the order of the columns in the baseline data
    :rtype: list[float]
    """
    model_wrapper = get_model_wrapper(task_type, target_column, baseline_data)

    baseline_explanations = compute_explanations(
        model_wrapper, baseline_data, categorical_features, target_column, task_type)
    log_time_and_message("Successfully computed explanations for dataset")

    return baseline_explanations


def write_to_mltable(explanations, dataset, file_path, categorical_features):
    """Write feature importance values to mltable.

    :param explanations: list of feature importances in the order of the baseline columns
    :type explanations: list[float]
    :param dataset: dataset to derive feature names
    :type dataset: pandas.Dataframe
    :param file_path: path to folder to save mltable
    :type file_path: string
    """
    log_time_and_message("Begin writing explanations to mltable")
    explanation_data = []
    for index in range(len(explanations)):
        dtype = constants.CATEGORICAL_FEATURE_CATEGORY if dataset.iloc[:, index].name in \
                        categorical_features else constants.NUMERICAL_FEATURE_CATEGORY
        new_row = pd.DataFrame(
                {constants.FEATURE_COLUMN: dataset.columns[index],
                 constants.METRIC_VALUE_COLUMN: explanations[index],
                 constants.METRIC_NAME_COLUMN: "FeatureImportance",
                 constants.FEATURE_CATEGORY_COLUMN: dtype,
                 constants.THRESHOLD_VALUE: float("nan")}, index=[0])
        explanation_data.append(new_row)
    row_count_data = pd.DataFrame(
                {constants.FEATURE_COLUMN: "",
                 constants.METRIC_VALUE_COLUMN: len(dataset.index),
                 constants.METRIC_NAME_COLUMN: constants.ROW_COUNT_COLUMN_NAME,
                 constants.FEATURE_CATEGORY_COLUMN: "",
                 constants.THRESHOLD_VALUE: float("nan")}, index=[0])
    explanation_data.append(row_count_data)

    metrics_data = pd.DataFrame(columns=[constants.FEATURE_COLUMN,
                                         constants.METRIC_VALUE_COLUMN,
                                         constants.METRIC_NAME_COLUMN,
                                         constants.FEATURE_CATEGORY_COLUMN,
                                         constants.THRESHOLD_VALUE], index=[0])
    print(explanation_data)
    metrics_data = pd.concat(explanation_data)
    spark_data = convert_pandas_to_spark(metrics_data)
    save_spark_df_as_mltable(spark_data, file_path)


def write_empty_signal_metrics_dataframe():
    """Write an empty feature importance metrics data frame."""
    spark = init_spark()
    metadata_schema = StructType(
        [
            StructField(constants.FEATURE_COLUMN, StringType(), True),
            StructField(constants.METRIC_VALUE_COLUMN, FloatType(), True),
            StructField(constants.METRIC_NAME_COLUMN, StringType(), True),
            StructField(constants.FEATURE_CATEGORY_COLUMN, StringType(), True),
            StructField(constants.THRESHOLD_VALUE, FloatType(), True),
        ]
    )
    # Create a new DataFrame with the metadata
    df = spark.createDataFrame([], metadata_schema)
    save_spark_df_as_mltable(df, args.signal_metrics)


def run(args):
    """Calculate feature importance."""
    try:
        # Check to see if target is present. If not, we'll return an empty dataframe
        if args.target_column is None:
            log_time_and_message("No target column given, creating an empty dataframe.")
            # Define a new schema for the DataFrame that will hold the metadata
            write_empty_signal_metrics_dataframe()
            return
        log_time_and_message("Reading data in spark and converting to pandas")
        baseline_df = try_read_mltable_in_spark_with_warning(args.baseline_data, "baseline_data")
        if not baseline_df:
            print("Skipping feature importance calculation.")
            return

        baseline_df = baseline_df.toPandas()
        task_type = args.task_type if args.task_type else determine_task_type(args.target_column, baseline_df)
        task_type = task_type.lower()
        log_time_and_message(f"Computed task type is {task_type}")

        categorical_features = compute_categorical_features(baseline_df, args.target_column)
        feature_importances = compute_feature_importance(
            task_type, args.target_column, baseline_df, categorical_features)
        feature_columns = baseline_df.drop([args.target_column], axis=1)
        write_to_mltable(feature_importances, feature_columns, args.signal_metrics, categorical_features)
        log_time_and_message("Successfully executed the feature importance component.")
    except Exception as e:
        log_time_and_message(f"Error encountered when executing feature importance component: {e}")
        raise e


if __name__ == "__main__":
    args = parse_args()

    run(args)
