# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for feature attribution drift component."""
import numpy as np
import pandas as pd
import logging

from responsibleai import RAIInsights
from sklearn.model_selection import train_test_split
from sklearn.metrics import ndcg_score
from ml_wrappers.model.predictions_wrapper import (
    PredictionsModelWrapperClassification,
    PredictionsModelWrapperRegression)

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
except ImportError:
    pass

from tabular.components.src._telemetry._loggerfactory import _LoggerFactory

_logger = logging.getLogger(__file__)
_ai_logger = None


def _get_logger():
    global _ai_logger
    if _ai_logger is None:
        _ai_logger = _LoggerFactory.get_logger(__file__)
    return _ai_logger


_get_logger()


def create_lightgbm_model(X, y, task_type):
    """Create model on which to calculate feature importances

    Args:
      X: list of x values (data excluding target columns)
      y: list y values (target column data)
      task_type: str, the task type (regression or classification) of the resulting model

    Returns:
       An appropriate model.
    """
    if(task_type == "classification"): 
         lgbm = LGBMClassifier(boosting_type='gbdt', learning_rate=0.1,
                          max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    else: 
        lgbm = LGBMRegressor(boosting_type='gbdt', learning_rate=0.1,
                         max_depth=5, n_estimators=200, n_jobs=1, random_state=777)
    model = lgbm.fit(X, y)
    
    _logger.info("Created lightgbm model")
    return model


def get_model_wrapper(task_type, target_column, baseline_df, production_df):
    """Create model wrapper using ml-wrappers on which to calculate feature importances
    
    Args:
      task_type: str, The task type (regression or classification) of the resulting model
      target_column: str, the column to predict
      baseline_df: The baseline dataframe
      production_df: The production dataframe

    Returns:
       An appropriate model wrapper
    """
    y_train = baseline_df[target_column]
    x_train = baseline_df.drop([target_column], axis=1)
    x_test = production_df.drop([target_column], axis=1)
    model = create_lightgbm_model(x_train, y_train, task_type)

    if task_type == 'classification':
        all_data = pd.concat([x_test, x_train])
        model_predict = model.predict(all_data)
        model_predict_proba = model.predict_proba(all_data)
        model_wrapper = PredictionsModelWrapperClassification(
            all_data,
            model_predict,
            model_predict_proba)
    else:
        all_data = pd.concat([x_test, x_train])
        model_predict = model.predict(all_data)
        model_wrapper = PredictionsModelWrapperRegression(all_data, model_predict)
    
    _logger.info("Created ml wrapper")
    return model_wrapper


def compute_categorical_features(baseline_df, target_column):
    """Compute which features are categorical based on data type of the columns.

    Args:
      baseline_df: The baseline dataframe
      target_column: str, the column to predict

    Returns:
      list: A list of categorical features.
    """
    categorical_features = []
    for column in baseline_df.columns:
        baseline_column = pd.Series(baseline_df[column])
        if baseline_column.dtype.name == 'object' and baseline_column.name != target_column:
            categorical_features.append(baseline_column.name)
    _logger.info("Successfully categorized columns")
    return categorical_features


def compute_explanations(model_wrapper, df, categorical_features, target_column, task_type):
    """Compute explanations (feature importances) for a given dataset

    Args:
      model_wrapper: wrapper around a model that can be used to calculate explanations
      df: The dataframe
      categorical_features: list of categorical features not including the target column
      target_column: str, the column to predict
      task_type: str, the task type (regression or classification) of the resulting model

    Returns:
      list: A list of explanations.
    """
    # Create the RAI Insights object, use either baseline as train and test data
    rai_i: RAIInsights = RAIInsights(
        model_wrapper, df, df, target_column, task_type, categorical_features=categorical_features
    )

    # Add the explanation and compute
    rai_i.explainer.add()
    rai_i.compute()

    return rai_i.explainer._explanation.global_importance_values


def calculate_attribution_drift(baseline_explanations, production_explanations):
    """Compute feature attribution drift given two sets of explanations

    Args:
      baseline_explanations: list of explanations calculated using the baseline dataframe
      production_explanations: list of explanations calculated using the production dataframe
      
    Returns:
      float: the ndcg metric between the baseline and production data
    """
    true_relevance = np.asarray([baseline_explanations])
    relevance_score = np.asarray([production_explanations])
    feature_attribution_drift = ndcg_score(true_relevance, relevance_score)
    print(feature_attribution_drift)
    return feature_attribution_drift


def compute_attribution_drift(task_type, target_column, baseline_df, production_df):
    """Compute feature attribution drift by calculating feature importances on each 
    dataframe input and using these to calculate the ndcg metric

    Args:
      task_type: str, the task type (regression or classification) of the resulting model
      target_column: str, the column to predict
      baseline_df: The baseline dataframe
      production_df: The production dataframe
      
    Returns:
      float: the ndcg metric between the baseline and production data
    """
    model_wrapper = get_model_wrapper(task_type, target_column, baseline_df, production_df)

    categorical_features = compute_categorical_features(baseline_df, target_column)

    baseline_explanations = compute_explanations(model_wrapper, baseline_df, categorical_features, target_column, task_type)
    _logger.info("Successfully computed explanations for baseline dataset")

    production_explanations = compute_explanations(model_wrapper, production_df, categorical_features, target_column, task_type)
    _logger.info("Successfully computed explanations for productiond dataset")

    return calculate_attribution_drift(baseline_explanations, production_explanations)