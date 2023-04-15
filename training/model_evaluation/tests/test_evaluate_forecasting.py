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


class TestEvaluateForecast(unittest.TestCase):
    """Test the evaluator on the forecasting tasks."""

    DATE = 'date'
    GRAIN = 'grain'
    TGT = 'y'
    PRED = 'y_pred'

    def _make_forecasting(self, grains, forecast_horizon=10):
        """Return X_train, X_test, y_pred"""
        dfs_train = []
        dfs_test = []
        for i in range(grains):
            X_one = pd.DataFrame({
                TestEvaluateForecast.DATE: pd.date_range('2001-01-01', freq='D', periods=40),
                TestEvaluateForecast.GRAIN: i,
                TestEvaluateForecast.TGT: i * 40 + np.arange(40, dtype='float')
            })
            dfs_train.append(X_one.iloc[:-forecast_horizon])
            dfs_test.append(X_one.iloc[-forecast_horizon:])
        X_train = pd.concat(dfs_train, sort=False, ignore_index=True)
        X_test = pd.concat(dfs_test, sort=False, ignore_index=True)
        np.random.seed(42)
        y_pred = X_test[TestEvaluateForecast.TGT].values + np.random.rand(X_test.shape[0])
        return X_train, X_test, y_pred

    def _calculate_rmse(self, X_train, X_test, y_test, y_pred, aggregate_fn):
        """Calculate the nrmse for predictions."""
        arr_rv = []
        X_test[TestEvaluateForecast.TGT] = y_test
        X_test[TestEvaluateForecast.PRED] = y_pred
        norm_ob = X_train.groupby(
            TestEvaluateForecast.GRAIN) if X_train is not None else None
        for ts, x_one in X_test.groupby(TestEvaluateForecast.GRAIN):
            rmse = np.sqrt(sklearn.metrics.mean_squared_error(
                x_one[TestEvaluateForecast.TGT].values,
                x_one[TestEvaluateForecast.PRED].values,
                multioutput='uniform_average'))
            norm_df = norm_ob.get_group(ts) if norm_ob is not None else x_one

            arr_rv.append(rmse / np.abs(norm_df[
                TestEvaluateForecast.TGT].max() - norm_df[
                    TestEvaluateForecast.TGT].min()))
        return aggregate_fn(arr_rv)

    def test_without_time_series_ids_mean(self):
        """Test data set without time series ids."""
        _, X_test, y_pred = self._make_forecasting(grains=1)
        y_test = X_test.pop(TestEvaluateForecast.TGT).values
        eval_results = evaluate_predictions(
            y_test, y_pred, y_pred_proba=None, task_type=TASK.FORECASTING,
            metrics_config={ForecastingConfigContract.TIME_COLUMN_NAME: TestEvaluateForecast.DATE},
            X_test=X_test)
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        self.assertIn("normalized_root_mean_squared_error", eval_results.metrics)
        self.assertAlmostEqual(eval_results.metrics["normalized_root_mean_squared_error"], expected_nrmse)

    def test_with_time_series_ids(self):
        """Test data set with time series ids."""
        _, X_test, y_pred = self._make_forecasting(grains=2)
        y_test = X_test.pop(TestEvaluateForecast.TGT).values
        eval_results = evaluate_predictions(
            y_test, y_pred, y_pred_proba=None, task_type=TASK.FORECASTING,
            metrics_config={
                ForecastingConfigContract.TIME_COLUMN_NAME: TestEvaluateForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestEvaluateForecast.GRAIN
            },
            X_test=X_test)
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        self.assertIn("normalized_root_mean_squared_error", eval_results.metrics)
        self.assertAlmostEqual(eval_results.metrics["normalized_root_mean_squared_error"], expected_nrmse)

    def _save_mltable(self, data, temp_dir, name):
        """Convenience method to save the mltable."""
        mltable_dir = os.path.join(temp_dir, name)
        os.makedirs(mltable_dir, exist_ok=True)
        parquet_name = os.path.join(mltable_dir, name + '.parquet').replace('\\\\', '/')
        data.to_parquet(parquet_name, index=False)
        parquet_paths = [{'file': './' + name + '.parquet'}]
        ml_table = mltable.from_parquet_files(parquet_paths)
        ml_table.save(mltable_dir)

    def _do_test_compute_component_e2e(self, config_as_string):
        """Test component end to end."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        Y_pred_table = pd.DataFrame({
            TestEvaluateForecast.PRED: y_pred
        })
        y_test = X_test.pop(TestEvaluateForecast.TGT).values
        expected_nrmse = self._calculate_rmse(None, X_test, y_test, y_pred, np.mean)
        X_test[TestEvaluateForecast.TGT] = y_test
        with tempfile.TemporaryDirectory() as d:
            self._save_mltable(X_test, d, 'x_test')
            input_test = os.path.join(d, 'x_test')
            self._save_mltable(Y_pred_table, d, 'y_pred')
            input_pred = os.path.join(d, 'y_pred')
            metrics_config_dt = {
                ForecastingConfigContract.TIME_COLUMN_NAME: TestEvaluateForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestEvaluateForecast.GRAIN
            }
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            param_list = [
                sys.executable, exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths_mltable", input_test,
                "--ground_truths_column_name", TestEvaluateForecast.TGT,
                "--predictions_mltable", input_pred,
                "--output", outputs
                ]
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
        self._do_test_compute_component_e2e(True)

    def test_compute_component_e2e_config_as_file(self):
        """Test evaluation if config is a string."""
        self._do_test_compute_component_e2e(False)

    def test_compute_component_e2e_errors(self):
        """Test component end to end."""
        exec_path = compute_metrics.__file__
        _, X_test, y_pred = self._make_forecasting(grains=2)
        Y_pred_table = pd.DataFrame({
            TestEvaluateForecast.PRED: y_pred
        })
        y_test = X_test.pop(TestEvaluateForecast.TGT).values
        X_test[TestEvaluateForecast.TGT] = y_test
        with tempfile.TemporaryDirectory() as d:
            self._save_mltable(X_test, d, 'x_test')
            input_test = os.path.join(d, 'x_test')
            self._save_mltable(Y_pred_table, d, 'y_pred')
            input_pred = os.path.join(d, 'y_pred')
            metrics_config = json.dumps({
                ForecastingConfigContract.TIME_COLUMN_NAME: TestEvaluateForecast.DATE,
                ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES: TestEvaluateForecast.GRAIN
            })
            outputs = os.path.join(d, 'outputs')
            os.makedirs(outputs, exist_ok=True)
            sys.argv = [
                exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths", input_test,
                "--ground_truths_column_name", TestEvaluateForecast.TGT,
                "--predictions_mltable", input_pred,
                "--output", outputs,
                "--config_str", metrics_config]
            with self.assertRaisesRegex(DataValidationException, 'For forecasting tasks,.+'):
                compute_metrics.test_component()
            sys.argv = [
                exec_path,
                "--task", TASK.FORECASTING,
                "--ground_truths_mltable", input_test,
                "--predictions_mltable", input_pred,
                "--output", outputs,
                "--config_str", metrics_config]
            with self.assertRaisesRegex(DataValidationException, 'For forecasting tasks,.+'):
                compute_metrics.test_component()


if __name__ == '__main__':
    unittest.main()
