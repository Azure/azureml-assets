# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for feature attribution drift component."""
import pandas as pd
import logging

from responsibleai import RAIInsights
from ml_wrappers.model.predictions_wrapper import (
    PredictionsModelWrapperClassification,
    PredictionsModelWrapperRegression)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    pass

_logger = logging.getLogger(__file__)
logging.basicConfig(level=logging.INFO)


def create_lightgbm_model(X, y, task_type):
    """Create model on which to calculate feature importances

      :param X: x values (data excluding target columns)
      :type X: pandas.Dataframe
      :param y: nparray of y values (target column data)
      :type y: nparray
      :param task_type: the task type (regression or classification) of the resulting model
      :type task_type: string
      :return: an appropriate model wrapper
      :rtype: LightGBMClassifier or LightGBMRegressor
    """
    if (task_type == "classification"):
        lgbm = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1,
                              max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    else:
        lgbm = LGBMRegressor(boosting_type='gbdt', learning_rate=0.1,
                             max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    model = lgbm.fit(X, y)

    _logger.info("Created lightgbm model using task_type: {0}".format(task_type))
    return model


def get_model_wrapper(task_type, target_column, baseline_dataframe, production_dataframe=None):
    """Create model wrapper using ml-wrappers on which to calculate feature importances

      :param task_type: The task type (regression or classification) of the resulting model
      :type task_type: string
      :param target_column: the column to predict
      :type target_column: string
      :param baseline_dataframe: The baseline data meaning the data used to create the
      model monitor
      :type baseline_dataframe: pandas.DataFrame
      :param production_dataframe: The production data meaning the most recent set of data
      sent to the model monitor, the current set of data
      :type production_dataframe: pandas.DataFrame
      :return: an appropriate model wrapper
      :rtype: PredictionsModelWrapperRegression or PredictionsModelWrapperClassification
    """
    y_train = baseline_dataframe[target_column]
    x_train = baseline_dataframe.drop([target_column], axis=1)
    model = create_lightgbm_model(x_train, y_train, task_type)
    if production_dataframe is not None:
        x_test = production_dataframe.drop([target_column], axis=1)
        all_data = pd.concat([x_test, x_train])
    else:
        all_data = x_train
    model_predict = model.predict(all_data)

    if task_type == 'classification':
        model_predict_proba = model.predict_proba(all_data)
        model_wrapper = PredictionsModelWrapperClassification(
            all_data,
            model_predict,
            model_predict_proba)
    else:
        model_wrapper = PredictionsModelWrapperRegression(all_data, model_predict)

    _logger.info("Created ml wrapper")
    return model_wrapper


def compute_categorical_features(baseline_dataframe, target_column):
    """Compute which features are categorical based on data type of the columns.

      :param baseline_dataframe: The baseline data meaning the data used to create the
      model monitor
      :type baseline_dataframe: pandas.DataFrame
      :param target_column: the column to predict
      :type target_column: string
      :return: categorical features
      :rtype: list[string]
    """
    categorical_features = []
    for column in baseline_dataframe.columns:
        baseline_column = pd.Series(baseline_dataframe[column])
        if baseline_column.dtype.name == 'object' and baseline_column.name != target_column:
            categorical_features.append(baseline_column.name)
    _logger.info("Successfully categorized columns")
    return categorical_features


def compute_explanations(model_wrapper, dataframe, categorical_features, target_column, task_type):
    """Compute explanations (feature importances) for a given dataset

      :param model_wrapper wrapper: around a model that can be used to calculate explanations
      :param  dataframe: The data used to calculate the explanations
      :type dataframe: pandas.DataFrame
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
    rai_i: RAIInsights = RAIInsights(
        model_wrapper, dataframe, dataframe, target_column, task_type, categorical_features=categorical_features
    )

    # Add the global explanations using batching to allow for larger input data sizes
    rai_i.explainer.add()
    evaluation_data = dataframe.drop([target_column], axis=1)
    explanationData = rai_i.explainer.request_explanations(local=False, data=evaluation_data)
    return explanationData.precomputedExplanations.globalFeatureImportance['scores']
