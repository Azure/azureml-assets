# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift Output Metrics component."""

import pytest
from typing import List
from pyspark.sql import Row

from model_monitor_output_signal_metrics.builder.metric_output_builder import MetricOutputBuilder


@pytest.mark.unit
class TestMetricOutputBuilder:
    """Test class for data drift output metrics component."""

    def test_metrics_with_1_level_groups(self):
        signal_metrics: List[Row] = [
            Row(
                group="group_1",
                group_pivot="Aggregate",
                metric_name="num_calls",
                metric_value=71.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_2",
                group_pivot="Aggregate",
                metric_name="num_calls",
                metric_value=129.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_1",
                group_pivot="Aggregate",
                metric_name="num_calls_with_status_code_429",
                metric_value=35.0,
                threshold_value=10.0,
            ),
            Row(
                group="group_2",
                group_pivot="Aggregate",
                metric_name="num_calls_with_status_code_429",
                metric_value=63.0,
                threshold_value=10.0,
            ),
        ]
        
        metric_output_builder = MetricOutputBuilder(signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()
        
        assert metrics_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                    },
                    "group_2": {
                        "value": 129.0,
                        "threshold": 100.0,
                    },
                },
            },
            "num_calls_with_status_code_429": {
                "groups": {
                    "group_1": {
                        "value": 35.0,
                        "threshold": 10.0,
                    },
                    "group_2": {
                        "value": 63.0,
                        "threshold": 10.0,
                    },
                },
            },
        }
    
    def test_metrics_with_2_level_groups(self):
        """Test generate meta file content for success scenario."""
        signal_metrics: List[Row] = [
            Row(
                group="group_1",
                group_pivot="user_A",
                metric_name="num_calls",
                metric_value=18.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_pivot="user_A",
                metric_name="num_calls",
                metric_value=32.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_pivot="user_C",
                metric_name="num_calls",
                metric_value=32.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_pivot="user_D",
                metric_name="num_calls",
                metric_value=33.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_pivot="user_B",
                metric_name="num_calls",
                metric_value=18.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_pivot="user_B",
                metric_name="num_calls",
                metric_value=32.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_pivot="user_D",
                metric_name="num_calls",
                metric_value=17.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_pivot="user_C",
                metric_name="num_calls",
                metric_value=18.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_pivot="user_D",
                metric_name="num_calls_with_status_code_429",
                metric_value=17.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_pivot="user_B",
                metric_name="num_calls_with_status_code_429",
                metric_value=11.0,
                threshold_value=None,
            ),
            Row(
                group="group_1",
                group_pivot="user_B",
                metric_name="num_calls_with_status_code_429",
                metric_value=18.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_pivot="user_D",
                metric_name="num_calls_with_status_code_429",
                metric_value=11.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_pivot="user_C",
                metric_name="num_calls_with_status_code_429",
                metric_value=21.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_pivot="user_A",
                metric_name="num_calls_with_status_code_429",
                metric_value=20.0,
                threshold_value=None,
            ),
        ]
        
        metric_output_builder = MetricOutputBuilder(signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()
        
        assert metrics_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "groups": {
                            "user_A": {
                                "value": 18.0,
                                "threshold": 50.0,
                            },
                            "user_B": {
                                "value": 18.0,
                                "threshold": 50.0,
                            },
                            "user_C": {
                                "value": 18.0,
                                "threshold": 50.0,
                            },
                            "user_D": {
                                "value": 17.0,
                                "threshold": 50.0,
                            },
                        },
                    },
                    "group_2": {
                        "groups": {
                            "user_A": {
                                "value": 32.0,
                                "threshold": 50.0,
                            },
                            "user_B": {
                                "value": 32.0,
                                "threshold": 50.0,
                            },
                            "user_C": {
                                "value": 32.0,
                                "threshold": 50.0,
                            },
                            "user_D": {
                                "value": 33.0,
                                "threshold": 50.0,
                            },
                        },
                    },
                }
            },
            "num_calls_with_status_code_429": {
                "groups": {
                    "group_1": {
                        "groups": {
                            "user_B": {
                                "value": 18.0,
                            },
                            "user_D": {
                                "value": 17.0,
                            },
                        },
                    },
                    "group_2": {
                        "groups": {
                            "user_A": {
                                "value": 20.0,
                            },
                            "user_B": {
                                "value": 11.0,
                            },
                            "user_C": {
                                "value": 21.0,
                            },
                            "user_D": {
                                "value": 11.0,
                            },
                        },
                    },
                },
            },
        }
