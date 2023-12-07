# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""AutoML Forecasting Inference component code."""
import argparse
import json
import os
import pickle

import azureml.automl.core.shared.constants as automl_constants
import numpy as np
import pandas as pd
from azureml._common._error_definition import AzureMLError, error_decorator
from azureml._common._error_definition.user_error import BadArgument
from azureml.automl.core.shared.forecasting_exception import ForecastingDataException
from azureml.core import Run

try:
    import torch  # noqa: F401

    _torch_present = True
except ImportError:
    _torch_present = False


class _ForecastModes:
    """Defines the forecast modes."""

    _RECURSIVE_FORECAST_MODE = "recursive"
    _ROLLING_FORECAST_MODE = "rolling"


# ------------------------------------------
# Constant values.
_FORECAST_ORIGIN_COLUMN_NAME = "automl_forecast_origin"
_PREDICTED_COLUMN_NAME = "automl_prediction"
_ACTUAL_COLUMN_NAME = "automl_actual"
_PI = "prediction_interval"
_PTH_FILE_POSTFIX = ".pth"
_PT_FILE_POSTFIX = ".pt"
_CSV_POSTFIX = ".csv"
_PARQ_POSTFIX = ".parquet"
_CONDA_YAML_FILE_NAME = "conda.yaml"
_DNN_FORECASTING_PKG_NAME = "azureml-contrib-automl-dnn-forecasting"


# ------------------------------------------
# Error handling.
class _InferenceComponentErrorStrings:
    """Error strings for inference component."""

    _QUANTILE_FMT_BAD = "Invalid quantiles string format, use comma separated string of float values, " \
                        "e.g. '0.1,0.5,0.9'."


@error_decorator(use_parent_error_code=True)
class _InvalidQuantileValueArgument(BadArgument):
    @property
    def message_format(self) -> str:
        return _InferenceComponentErrorStrings._QUANTILE_FMT_BAD


_INVALID_ARG_QUANTILE_FORMAT = 'abc26505-87e4-4877-a855-7532473290a1'


# ------------------------------------------
# Util methods.
def _map_location_cuda(storage, loc):
    return storage.cuda()


def _get_test_data(target_column_name, test_dataset):
    print("Loading test data.")
    test_df = None
    for fle in filter(lambda x: x.endswith(_CSV_POSTFIX) or x.endswith(_PARQ_POSTFIX), os.listdir(test_dataset)):
        fl_path = os.path.join(test_dataset, fle)
        if fle.endswith(_CSV_POSTFIX):
            test_df = pd.read_csv(fl_path)
        else:
            test_df = pd.read_parquet(fl_path)
        # We read the first csv or parquet file as test data, in case there are more than one files in the folder.
        print(f"[DataLoading]The test data has been loaded from the file: [{fl_path}]")
        print(f"[DataLoading]Shape of the loaded test data: {test_df.shape}")
        break

    if test_df is None:
        raise ValueError("The test data set can not be found.")

    if target_column_name not in test_df:
        # The user doesn't provide actuals, we generate nan values for the y_test.
        print("The target column of actuals is not provided, generate the column with nan values.")
        y_test = np.full(test_df.shape[0], np.nan)
    else:
        y_test = test_df.pop(target_column_name).values

    print("Test data loading succeeded.")
    return test_df, y_test


def _get_model_fullpath(model_path):
    model_fl_name = ""
    for filename in os.listdir(model_path):
        if filename == automl_constants.MODEL_FILENAME or filename == automl_constants.PT_MODEL_FILENAME:
            model_fl_name = filename
            break

    model_full_path = ""
    if model_fl_name:
        model_full_path = os.path.join(model_path, model_fl_name)
    else:
        # Find the .pt or .pth pytorch model file in sub-folders:
        for root, dirs, files in os.walk(model_path):
            for filename in files:
                if filename.endswith(_PT_FILE_POSTFIX) or filename.endswith(_PTH_FILE_POSTFIX):
                    model_full_path = os.path.join(root, filename)
                    break

    if not model_full_path:
        raise Exception(f"Unable to find any valid model in folder {model_path}!")

    return model_full_path


def _get_model(model_full_path):
    fitted_model = None
    print(f"Loading the model from path: {model_full_path}")
    if model_full_path.endswith(_PT_FILE_POSTFIX) or model_full_path.endswith(_PTH_FILE_POSTFIX):
        if not _torch_present:
            raise Exception("Loading Forecasting TCN model requires torch to be installed in the environment.")

        if torch.cuda.is_available():
            map_location = _map_location_cuda
        else:
            map_location = "cpu"
        with open(model_full_path, "rb") as fh:
            fitted_model = torch.load(fh, map_location=map_location)
    else:
        # Load the sklearn pipeline.
        with open(model_full_path, 'rb') as fp:
            fitted_model = pickle.load(fp)

    print("Model loading succeeded.")
    return fitted_model


# ------------------------------------------
# Forecasting methods.
def _recursive_forecast(X_test, model, y_test):
    X_test_agg, y_test = model.preaggregate_data_set(X_test, y_test)
    y_pred, df_all = model.forecast(X_test, ignore_data_errors=True)
    df_all[_PREDICTED_COLUMN_NAME] = y_pred
    df_all[_ACTUAL_COLUMN_NAME] = y_test
    df_all.reset_index(inplace=True, drop=False)
    kept_columns = [model.time_column_name] + model.grain_column_names + [
        _PREDICTED_COLUMN_NAME, _ACTUAL_COLUMN_NAME
    ]
    df_all = df_all[kept_columns]
    return df_all


def _rolling_forecast(X_test, df_all, forecast_step, model, y_test):
    df_all = model.rolling_forecast(
        X_test, y_test, step=forecast_step, ignore_data_errors=True
    )
    # Add predictions, actuals, and horizon relative to rolling origin to the test feature data
    assign_dict = {
        model.forecast_origin_column_name: _FORECAST_ORIGIN_COLUMN_NAME,
        model.forecast_column_name: _PREDICTED_COLUMN_NAME,
        model.actual_column_name: _ACTUAL_COLUMN_NAME,
    }
    df_all.rename(columns=assign_dict, inplace=True)
    # Drop rows where prediction or actuals are nan,
    # this happens because of missing actuals or at edges of time due to lags/rolling windows.
    df_all.dropna(inplace=True)
    return df_all


def _get_quantiles(forecast_quantiles):
    quantiles = None
    if not forecast_quantiles:
        # Set the default quantile value.
        quantiles = [0.5]
    else:
        try:
            tmp = forecast_quantiles.strip("'\"")
            quantiles = [float(v) for v in tmp.split(",")]
        except Exception:
            raise ForecastingDataException._with_error(
                AzureMLError.create(
                    _InvalidQuantileValueArgument,
                    target='DemandForecastInference_quantile_forecast',
                    reference_code=_INVALID_ARG_QUANTILE_FORMAT
                )
            )

        if 0.5 not in quantiles:
            # Enforce the median in quantiles.
            quantiles.append(0.5)
        quantiles.sort()
    return quantiles


def _quantile_forecast(X_test, forecast_quantiles, model, y_test):
    quantiles = _get_quantiles(forecast_quantiles)
    model.quantiles = quantiles
    pred_quantiles = model.forecast_quantiles(X_test, ignore_data_errors=True)
    pred_quantiles[_PI] = pred_quantiles[[min(quantiles), max(quantiles)]].apply(
        lambda x: "[{}, {}]".format(x[0], x[1]), axis=1
    )
    y_pred = pred_quantiles[0.5].values

    df_all = X_test
    df_all[_PI] = pred_quantiles[_PI]
    df_all[_PREDICTED_COLUMN_NAME] = y_pred
    df_all[_ACTUAL_COLUMN_NAME] = y_test

    # Drop rows where prediction or actuals are nan,
    # this happens because of missing actuals or at edges of time due to lags/rolling windows.
    df_all = df_all[
        df_all[[_ACTUAL_COLUMN_NAME, _PREDICTED_COLUMN_NAME]].notnull().all(axis=1)
    ]
    kept_columns = [model.time_column_name] + model.grain_column_names + [
        _PI, _PREDICTED_COLUMN_NAME, _ACTUAL_COLUMN_NAME,
    ]
    df_all = df_all[kept_columns]

    return df_all


def _infer_forecasting_dataset(
    X_test, y_test, model, forecast_mode, forecast_step, forecast_quantiles,
    inference_output_file_name, evaluation_config_output_file_name
):
    print("Start the inference.")
    df_all = None
    if forecast_mode == _ForecastModes._RECURSIVE_FORECAST_MODE:
        # For recursive forecast mode, in the forecast data we are not guaranteed to have the same
        # dimension of output data as the input, so we have to pre-aggregate the data here.
        if not forecast_quantiles:
            df_all = _recursive_forecast(X_test, model, y_test)
        else:
            # If the user provides the quantile values, we support quantile forecast for recursive mode.
            df_all = _quantile_forecast(X_test, forecast_quantiles, model, y_test)
    elif forecast_mode == _ForecastModes._ROLLING_FORECAST_MODE:
        df_all = _rolling_forecast(X_test, df_all, forecast_step, model, y_test)
    else:
        # Invalid forecast mode, the code should not go here.
        raise Exception("Invalid forecast mode.")

    # Write the inference result file.
    print(f"Writing the inference result to path: {inference_output_file_name}")
    df_all.to_json(inference_output_file_name, orient='records', lines=True)

    # Write the evaluation config json file.
    print(f"Writing the evaluation config to path: {evaluation_config_output_file_name}")
    with open(evaluation_config_output_file_name, 'w') as fp:
        if forecast_mode == _ForecastModes._ROLLING_FORECAST_MODE:
            json.dump({
                'time_column_name': model.time_column_name,
                'time_series_id_column_names': model.grain_column_names,
                'forecast_origin_column_name': _FORECAST_ORIGIN_COLUMN_NAME,
                'predictions_column_name': _PREDICTED_COLUMN_NAME,
                'ground_truths_column_name': _ACTUAL_COLUMN_NAME
            }, fp)
        else:
            json.dump({
                'time_column_name': model.time_column_name,
                'time_series_id_column_names': model.grain_column_names,
                'predictions_column_name': _PREDICTED_COLUMN_NAME,
                'ground_truths_column_name': _ACTUAL_COLUMN_NAME
            }, fp)

    print("Inference succeeded.")


def _get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_data",
        type=str,
        dest="test_data",
        default="results",
        help="The test dataset path",
    )
    parser.add_argument(
        "--model_path", type=str, dest="model_path", help="Model to be loaded"
    )
    parser.add_argument(
        "--target_column_name",
        type=str,
        dest="target_column_name",
        help="The target column name.",
    )
    parser.add_argument(
        "--forecast_mode",
        type=str,
        dest="forecast_mode",
        default=_ForecastModes._RECURSIVE_FORECAST_MODE,
        help="Forecast mode",
    )
    parser.add_argument(
        "--forecast_step",
        type=int,
        dest="forecast_step",
        default="1",
        help="Number of steps of rolling forecast",
    )
    parser.add_argument(
        "--forecast_quantiles",
        type=str,
        dest="forecast_quantiles",
        default="",
        help="Comma separated quantile values for quantile forecast",
    )
    parser.add_argument(
        "--inference_output_file_name",
        type=str,
        dest="inference_output_file_name",
        default="inference_results",
        help="File name of the inference output",
    )
    parser.add_argument(
        "--evaluation_config_output_file_name",
        type=str,
        dest="evaluation_config_output_file_name",
        default="evaluation_config",
        help="File name of the output file of evaluation config",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    run = Run.get_context()

    args = _get_args()
    model_path = args.model_path
    test_data = args.test_data
    target_column_name = args.target_column_name
    forecast_mode = args.forecast_mode
    forecast_step = args.forecast_step
    forecast_quantiles = args.forecast_quantiles
    inference_output_file_name = args.inference_output_file_name
    evaluation_config_output_file_name = args.evaluation_config_output_file_name

    print("args passed in: ")
    print(f"model_path: {model_path}")
    print(f"test_data: {test_data}")
    print(f"target_column_name: {target_column_name}")
    print(f"forecast_mode: {forecast_mode}")
    print(f"forecast_step: {forecast_step}")
    print(f"forecast_quantiles: {forecast_quantiles}")
    print(f"inference_output_file_name: {inference_output_file_name}")
    print(f"evaluation_config_output_file_name: {evaluation_config_output_file_name}")

    model_full_path = _get_model_fullpath(model_path)
    fitted_model = _get_model(model_full_path)
    X_test_df, y_test = _get_test_data(target_column_name, test_data)

    _infer_forecasting_dataset(
        X_test_df, y_test, fitted_model, forecast_mode, forecast_step, forecast_quantiles,
        inference_output_file_name, evaluation_config_output_file_name
    )
