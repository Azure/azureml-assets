# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Test forecasting evaluation."""
import unittest

import json

import mltable
import numpy as np
import os
import pandas as pd
import sklearn.metrics
import subprocess
import sys
import tempfile

import compute_metrics

from utils import evaluate_predictions
from constants import TASK, ForecastingConfigContract
from exceptions import DataValidationException
from pandas.tseries.frequencies import to_offset


class TestComputeForecast(unittest.TestCase):
    """Test the evaluator on the forecasting tasks."""

    DATE = 'date'
    GRAIN = 'grain'
    TGT = 'y'
    PRED = 'y_pred'

    def _make_forecasting(self, grains, forecast_horizon=10):
        """Return X_train, X_test, y_pred."""
        dfs_train = []
        dfs_test = []
        for i in range(grains):
            X_one = pd.DataFrame({
                TestComputeForecast.DATE: pd.date_range('2001-01-01', freq='D', periods=40),
                TestComputeForecast.GRAIN: i,
                TestComputeForecast.TGT: i * 40 + np.arange(40, dtype='float')
            })
            dfs_train.append(X_one.iloc[:-forecast_horizon])
            dfs_test.append(X_one.iloc[-forecast_horizon:])
        X_train = pd.concat(dfs_train, sort=False, ignore_index=True)
        X_test = pd.concat(dfs_test, sort=False, ignore_index=True)
        np.random.seed(42)
        y_pred = X_test[TestComputeForecast.TGT].values + np.random.rand(X_test.shape[0])
        return X_train, X_test, y_pred

    def _calculate_rmse(self, X_train, X_test, y_test, y_pred, aggregate_fn):
        """Calculate the nrmse for predictions."""
        arr_rv = []
        X_test[TestComputeForecast.TGT] = y_test
        X_test[TestComputeForecast.PRED] = y_pred
        norm_ob = X_train.groupby(
            TestComputeForecast.GRAIN) if X_train is not None else None
        for ts, x_one in X_test.groupby(TestComputeForecast.GRAIN):
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(
                x_one[TestComputeForecast.TGT].values,
                x_one[TestComputeForecast.PRED].values,
                multioutput='uniform_average'))
            norm_df = norm_ob.get_group(ts) if norm_ob is not None else x_one

            arr_rv.append(rmse / np.abs(norm_df[
                TestComputeForecast.TGT].max() - norm_df[
                    TestComputeForecast.TGT].min()))
        return aggregate_fn(arr_rv)

    def test_without_time_series_ids_mean(self):
        """Test data set without time series ids."""
        _, X_test, y_pred = self._make_forecasting(grains=1)
        y_test = X_test.pop(TestComputeForecast.TGT).values
        eval_results = evaluate_predictions(
            y_test, y_pred, y_pred_proba=None, task_type=TASK.FORECASTING,
            metrics_config={ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE},
            X_test=X_test)
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        self.assertIn("normalized_root_mean_squared_error", eval_results.metrics)
        self.assertAlmostEqual(eval_results.metrics["normalized_root_mean_squared_error"], expected_nrmse)

    def test_with_time_series_ids(self):
        """Test data set with time series ids."""
        _, X_test, y_pred = self._make_forecasting(grains=2)
        y_test = X_test.pop(TestComputeForecast.TGT).values
        eval_results = evaluate_predictions(
            y_test, y_pred, y_pred_proba=None, task_type=TASK.FORECASTING,
            metrics_config={
                ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestComputeForecast.GRAIN
            },
            X_test=X_test)
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        self.assertIn("normalized_root_mean_squared_error", eval_results.metrics)
        self.assertAlmostEqual(eval_results.metrics["normalized_root_mean_squared_error"], expected_nrmse)

    def _save_mltable(self, data, temp_dir, name):
        """Save the mltable."""
        mltable_dir = os.path.join(temp_dir, name)
        os.makedirs(mltable_dir, exist_ok=True)
        parquet_name = os.path.join(mltable_dir, name + '.parquet').replace('\\\\', '/')
        data.to_parquet(parquet_name, index=False)
        parquet_paths = [{'file': './' + name + '.parquet'}]
        ml_table = mltable.from_parquet_files(parquet_paths)
        ml_table.save(mltable_dir)

    def _do_test_compute_component_e2e(self, config_as_string, columns_in_config):
        """Test component end to end."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        Y_pred_table = pd.DataFrame({
            TestComputeForecast.PRED: y_pred
        })
        y_test = X_test.pop(TestComputeForecast.TGT).values
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        X_test[TestComputeForecast.TGT] = y_test
        with tempfile.TemporaryDirectory() as d:
            self._save_mltable(X_test, d, 'x_test')
            input_test = os.path.join(d, 'x_test')
            self._save_mltable(Y_pred_table, d, 'y_pred')
            input_pred = os.path.join(d, 'y_pred')
            metrics_config_dt = {
                ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestComputeForecast.GRAIN
            }
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            param_list = [
                sys.executable, exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths_mltable", input_test,
                "--predictions_mltable", input_pred,
                "--output", outputs
            ]
            if columns_in_config:
                metrics_config_dt['ground_truths_column_name'] = TestComputeForecast.TGT
            else:
                param_list.append('--ground_truths_column_name')
                param_list.append(TestComputeForecast.TGT)
            if config_as_string:
                param_list.append("--config_str")
                metrics_config = json.dumps(metrics_config_dt)
            else:
                param_list.append("--config-file-name")
                metrics_config = os.path.join(d, 'config.json')
                with open(metrics_config, 'w') as cn:
                    json.dump(metrics_config_dt, cn)
            param_list.append(metrics_config)
            subprocess.run(param_list)
            with open(os.path.join(outputs, 'evaluationResult', 'metrics.json'), 'r') as f:
                eval_results = json.load(f)
            self.assertIn("normalized_root_mean_squared_error", eval_results)
            self.assertAlmostEqual(eval_results["normalized_root_mean_squared_error"], expected_nrmse)

    def test_compute_component_e2e_config_as_str(self):
        """Test evaluation if config is a string."""
        self._do_test_compute_component_e2e(True, False)

    def test_compute_component_e2e_config_as_file(self):
        """Test evaluation if config is a string."""
        self._do_test_compute_component_e2e(False, False)

    def test_compute_component_e2e_config_as_str_column_conf(self):
        """Test evaluation if config is a string."""
        self._do_test_compute_component_e2e(True, True)

    def test_compute_component_e2e_config_as_file_column_conf(self):
        """Test evaluation if config is a string."""
        self._do_test_compute_component_e2e(False, True)

    def test_compute_component_e2e_errors(self):
        """Test component end to end."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        Y_pred_table = pd.DataFrame({
            TestComputeForecast.PRED: y_pred
        })
        y_test = X_test.pop(TestComputeForecast.TGT).values
        X_test[TestComputeForecast.TGT] = y_test
        with tempfile.TemporaryDirectory() as d:
            self._save_mltable(X_test, d, 'x_test')
            input_test = os.path.join(d, 'x_test')
            self._save_mltable(Y_pred_table, d, 'y_pred')
            input_pred = os.path.join(d, 'y_pred')
            metrics_config = json.dumps({
                ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestComputeForecast.GRAIN
            })
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            sys.argv = [
                exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths_mltable", input_test,
                "--predictions_mltable", input_pred,
                "--output", outputs,
                "--config_str", metrics_config]
            with self.assertRaisesRegex(DataValidationException, 'For forecasting tasks,.+'):
                compute_metrics.test_component()

    def _do_test_prediction_column_names(self, columns_in_config, config_as_string):
        """Test data set with prediction column names."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        X_test['predictions'] = y_pred
        y_test = X_test.pop(TestComputeForecast.TGT).values
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        X_test[TestComputeForecast.TGT] = y_test
        with tempfile.TemporaryDirectory() as d:
            self._save_mltable(X_test, d, 'x_test')
            input_test = os.path.join(d, 'x_test')
            metrics_config_dt = {
                ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestComputeForecast.GRAIN
            }
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            param_list = [
                sys.executable, exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths_mltable", input_test,
                "--predictions_mltable", input_test,
                "--output", outputs
            ]
            if columns_in_config:
                metrics_config_dt['ground_truths_column_name'] = TestComputeForecast.TGT
                metrics_config_dt['predictions_column_name'] = 'predictions'
            else:
                param_list.append('--ground_truths_column_name')
                param_list.append(TestComputeForecast.TGT)
                param_list.append('--predictions_column_name')
                param_list.append('predictions')
            if config_as_string:
                param_list.append("--config_str")
                metrics_config = json.dumps(metrics_config_dt)
            else:
                param_list.append("--config-file-name")
                metrics_config = os.path.join(d, 'config.json')
                with open(metrics_config, 'w') as cn:
                    json.dump(metrics_config_dt, cn)
            param_list.append(metrics_config)
            subprocess.run(param_list)
            with open(os.path.join(outputs, 'evaluationResult', 'metrics.json'), 'r') as f:
                eval_results = json.load(f)
            self.assertIn("normalized_root_mean_squared_error", eval_results)
            self.assertAlmostEqual(eval_results["normalized_root_mean_squared_error"], expected_nrmse)

    def test_e2e_prediction_column_names_in_config(self):
        """Test the prediction when the prediction column name is provided in config."""
        self._do_test_prediction_column_names(True, True)
        self._do_test_prediction_column_names(True, False)

    def test_e2e_prediction_column_names_as_single_param(self):
        """Test the prediction when the prediction column name is provided as single parameter."""
        self._do_test_prediction_column_names(False, True)
        self._do_test_prediction_column_names(False, False)

    def _do_test_forecast_as_jsonl(self, columns_in_config, config_as_string):
        """Test forecast on files in jsonl format."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        X_test['predictions'] = y_pred
        y_test = X_test.pop(TestComputeForecast.TGT).values
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        X_test[TestComputeForecast.TGT] = y_test
        with tempfile.TemporaryDirectory() as d:
            input_test = os.path.join(d, 'x_test.jsonl')
            X_test.to_json(os.path.join(d, 'x_test.jsonl'), orient='records', lines=True)

            metrics_config_dt = {
                ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestComputeForecast.GRAIN
            }
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            param_list = [
                sys.executable, exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths", input_test,
                "--predictions", input_test,
                "--output", outputs
            ]
            if columns_in_config:
                metrics_config_dt['ground_truths_column_name'] = TestComputeForecast.TGT
                metrics_config_dt['predictions_column_name'] = 'predictions'
            else:
                param_list.append('--ground_truths_column_name')
                param_list.append(TestComputeForecast.TGT)
                param_list.append('--predictions_column_name')
                param_list.append('predictions')
            if config_as_string:
                param_list.append("--config_str")
                metrics_config = json.dumps(metrics_config_dt)
            else:
                param_list.append("--config-file-name")
                metrics_config = os.path.join(d, 'config.json')
                with open(metrics_config, 'w') as cn:
                    json.dump(metrics_config_dt, cn)
            param_list.append(metrics_config)
            subprocess.run(param_list)
            with open(os.path.join(outputs, 'evaluationResult', 'metrics.json'), 'r') as f:
                eval_results = json.load(f)
            self.assertIn("normalized_root_mean_squared_error", eval_results)
            self.assertAlmostEqual(eval_results["normalized_root_mean_squared_error"], expected_nrmse)

    def test_forecast_as_jsonl_config(self):
        """
        Test the forecast on jsonl files.

        Ground true and prediction columns are provided in metrics config.
        """
        self._do_test_forecast_as_jsonl(True, True)
        self._do_test_forecast_as_jsonl(True, False)

    def test_forecast_as_jsonl(self):
        """
        Test the forecast on jsonl files.

        Ground true and prediction columns are as separate parameters.
        """
        self._do_test_forecast_as_jsonl(False, True)
        self._do_test_forecast_as_jsonl(False, False)

    def _do_test_rolling_forecast(self, config_as_string, origin_in_config):
        """Do the actual test."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        X_test['predictions'] = y_pred
        fc_ori_column = 'forecast_origin' if origin_in_config else '_automl_forecast_origin'
        freq = to_offset('D')
        X_test_new = X_test.copy()
        early_origin = X_test[TestComputeForecast.DATE].min() - freq
        late_origin = X_test[TestComputeForecast.DATE].min()
        X_test_new[fc_ori_column] = early_origin
        X_test[fc_ori_column] = late_origin
        X_test = pd.concat([X_test_new, X_test], sort=False, ignore_index=True)
        with tempfile.TemporaryDirectory() as d:
            input_test = os.path.join(d, 'x_test.jsonl')
            X_test.to_json(os.path.join(d, 'x_test.jsonl'), orient='records', lines=True)

            metrics_config_dt = {
                ForecastingConfigContract.TIME_COLUMN_NAME: TestComputeForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestComputeForecast.GRAIN,
                'ground_truths_column_name': TestComputeForecast.TGT,
                'predictions_column_name': 'predictions'
            }
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            sys.argv = [
                exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths", input_test,
                "--predictions", input_test,
                "--output", outputs
            ]
            if origin_in_config:
                metrics_config_dt[ForecastingConfigContract.FORECAST_ORIGIN_COLUMN_NAME] = fc_ori_column
            if config_as_string:
                sys.argv.append("--config_str")
                metrics_config = json.dumps(metrics_config_dt)
            else:
                sys.argv.append("--config-file-name")
                metrics_config = os.path.join(d, 'config.json')
                with open(metrics_config, 'w') as cn:
                    json.dump(metrics_config_dt, cn)
            sys.argv.append(metrics_config)
            compute_metrics.test_component()
            with open(os.path.join(
                    outputs, 'evaluationResult', 'artifacts', 'forecast_time_series_id_distribution_table'),
                    'r') as f:
                eval_results = json.load(f)
            distribution_table = pd.DataFrame(eval_results['data'])
            self.assertEqual(distribution_table.shape[0], 4)
            self.assertIn(TestComputeForecast.GRAIN, distribution_table.columns)
            self.assertIn(fc_ori_column, distribution_table.columns)
            self.assertIn(early_origin.isoformat(), distribution_table[fc_ori_column].values)
            self.assertIn(late_origin.isoformat(), distribution_table[fc_ori_column].values)

    def test_distribution_table_with_forecast_origin_column_config(self):
        """Test forecast origin column in the config."""
        self._do_test_rolling_forecast(True, True)
        self._do_test_rolling_forecast(False, True)

    def test_distribution_table_without_forecast_origin_column_config(self):
        """Test forecast origin column in the config."""
        self._do_test_rolling_forecast(True, False)
        self._do_test_rolling_forecast(False, False)


if __name__ == '__main__':
    unittest.main()
