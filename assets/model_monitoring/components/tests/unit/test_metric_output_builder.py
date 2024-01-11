# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the Data Drift Output Metrics component."""

import pytest
from typing import List
from pyspark.sql import Row
import numpy as np

from model_monitor_metric_outputter.builder.metric_output_builder import MetricOutputBuilder


@pytest.mark.unit1
class TestMetricOutputBuilder:
    """Test class for metric output builder."""

    def test_metrics_with_none_values(self, mock_runmetric_client, monitor_name, signal_name):
        """Test metrics output builder metrics with none."""
        signal_metrics: List[Row] = [
            Row(
                metric_name="num_calls",
                metric_value=0,
                threshold_value=100.0,
            ),
            Row(
                metric_name="num_calls_2",
                metric_value=None,
                threshold_value=100.0,
            ),
            Row(
                metric_name="num_calls_with_status_code_429",
                metric_value=35.0,
                threshold_value=0,
            ),
            Row(
                metric_name="num_calls_with_status_code_500",
                metric_value=63.0,
                threshold_value=None,
            ),
            Row(
                metric_name="num_calls_with_status_code_501",
            ),
        ]
        metric_output_builder = MetricOutputBuilder(mock_runmetric_client, monitor_name, signal_name, signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()
        assert metrics_dict == {
            "num_calls": {
                "threshold": 100.0,
                "timeseries": {
                    "metricNames": {
                        "threshold": "threshold",
                        "value": "value"
                    },
                    "runId": mock_runmetric_client.run_id,
                    },
                "value": 0
                },
            "num_calls_2": {
                "threshold": 100.0,
                "timeseries": {
                    "metricNames": {
                        "threshold": "threshold",
                        "value": "value"
                    },
                    "runId": mock_runmetric_client.run_id,
                    },
                },
            "num_calls_with_status_code_429": {
                "threshold": 0,
                "timeseries": {
                    "metricNames": {
                        "threshold": "threshold",
                        "value": "value"
                    },
                    "runId": mock_runmetric_client.run_id,
                    },
                "value": 35.0
                },
            "num_calls_with_status_code_500": {
                "timeseries": {
                    "metricNames": {
                        "threshold": "threshold",
                        "value": "value"
                    },
                    "runId": mock_runmetric_client.run_id,
                    },
                "value": 63.0
                },
        }

    def test_metrics_with_nan_values(self, mock_runmetric_client, monitor_name, signal_name):
        """Test metrics output builder metrics with nan."""
        signal_metrics: List[Row] = [
            Row(
                metric_name="num_calls",
                metric_value=float("Nan")
            ),
        ]
        metric_output_builder = MetricOutputBuilder(mock_runmetric_client, monitor_name, signal_name, signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()
        assert np.isnan(metrics_dict["num_calls"]["value"])

    def test_metrics_with_1_level_groups(self, mock_runmetric_client, monitor_name, signal_name):
        """Test metrics output builder for metrics with one level metric groups."""
        signal_metrics: List[Row] = [
            Row(
                group="group_1",
                group_dimension="Aggregate",
                metric_name="num_calls",
                metric_value=71.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_2",
                group_dimension="Aggregate",
                metric_name="num_calls",
                metric_value=129.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_1",
                group_dimension="Aggregate",
                metric_name="num_calls_with_status_code_429",
                metric_value=35.0,
                threshold_value=10.0,
            ),
            Row(
                group="group_2",
                group_dimension="Aggregate",
                metric_name="num_calls_with_status_code_429",
                metric_value=63.0,
                threshold_value=10.0,
            ),
        ]

        metric_output_builder = MetricOutputBuilder(mock_runmetric_client, monitor_name, signal_name, signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()

        assert metrics_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                    "group_2": {
                        "value": 129.0,
                        "threshold": 100.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                },
            },
            "num_calls_with_status_code_429": {
                "groups": {
                    "group_1": {
                        "value": 35.0,
                        "threshold": 10.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                    "group_2": {
                        "value": 63.0,
                        "threshold": 10.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                },
            },
        }

    def test_empty_groups_and_group_dimensions_success(self, mock_runmetric_client, monitor_name, signal_name):
        """Test metrics output builder for metrics that does not contain groups and group dimensions."""
        signal_metrics: List[Row] = [
            Row(
                group="group_1",
                group_dimension="Aggregate",
                metric_name="num_calls",
                metric_value=71.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_2",
                group_dimension="Aggregate",
                metric_name="num_calls",
                metric_value=129.0,
                threshold_value=100.0,
            ),
            Row(
                group="",
                metric_name="num_calls_with_status_code_429",
                metric_value=35.0,
                threshold_value=10.0,
            ),
            Row(
                metric_name="num_calls_with_status_code_500",
                metric_value=63.0,
                threshold_value=10.0,
            ),
            Row(
                group=None,
                metric_name="num_calls_with_status_code_504",
                metric_value=22.0,
                threshold_value=10.0,
            ),
        ]

        metric_output_builder = MetricOutputBuilder(mock_runmetric_client, monitor_name, signal_name, signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()
        assert metrics_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                    "group_2": {
                        "value": 129.0,
                        "threshold": 100.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                },
            },
            "num_calls_with_status_code_429": {
                "value": 35.0,
                "threshold": 10.0,
                "timeseries": {
                    "runId": mock_runmetric_client.run_id,
                    "metricNames": {
                        "value": "value",
                        "threshold": "threshold"
                    }
                }
            },
            "num_calls_with_status_code_500": {
                "value": 63.0,
                "threshold": 10.0,
                "timeseries": {
                    "runId": mock_runmetric_client.run_id,
                    "metricNames": {
                        "value": "value",
                        "threshold": "threshold"
                    }
                }
            },
            "num_calls_with_status_code_504": {
                "value": 22.0,
                "threshold": 10.0,
                "timeseries": {
                    "runId": mock_runmetric_client.run_id,
                    "metricNames": {
                        "value": "value",
                        "threshold": "threshold"
                    }
                }
            }
        }

    def test_group_without_group_dimension_success(self, mock_runmetric_client, monitor_name, signal_name):
        """Test metrics output builder for metrics that contains group but not group dimension."""
        signal_metrics: List[Row] = [
            Row(
                group="group_1",
                group_dimension="",
                metric_name="num_calls",
                metric_value=71.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_2",
                group_dimension=None,
                metric_name="num_calls",
                metric_value=129.0,
                threshold_value=100.0,
            ),
            Row(
                group="group_1",
                metric_name="num_calls_with_status_code_429",
                metric_value=35.0,
                threshold_value=10.0,
            ),
            Row(
                group="group_2",
                metric_name="num_calls_with_status_code_429",
                metric_value=63.0,
                threshold_value=10.0,
            ),
        ]

        metric_output_builder = MetricOutputBuilder(mock_runmetric_client, monitor_name, signal_name, signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()
        assert metrics_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "value": 71.0,
                        "threshold": 100.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                    "group_2": {
                        "value": 129.0,
                        "threshold": 100.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                },
            },
            "num_calls_with_status_code_429": {
                "groups": {
                    "group_1": {
                        "value": 35.0,
                        "threshold": 10.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                    "group_2": {
                        "value": 63.0,
                        "threshold": 10.0,
                        "timeseries": {
                            "runId": mock_runmetric_client.run_id,
                            "metricNames": {
                                "value": "value",
                                "threshold": "threshold"
                            }
                        }
                    },
                },
            },
        }

    def test_metrics_with_2_level_groups(self, mock_runmetric_client, monitor_name, signal_name):
        """Test metrics output builder for metrics with two level metric groups."""
        signal_metrics: List[Row] = [
            Row(
                group="group_1",
                group_dimension="user_A",
                metric_name="num_calls",
                metric_value=18.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_dimension="user_A",
                metric_name="num_calls",
                metric_value=32.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_dimension="user_C",
                metric_name="num_calls",
                metric_value=32.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_dimension="user_D",
                metric_name="num_calls",
                metric_value=33.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_dimension="user_B",
                metric_name="num_calls",
                metric_value=18.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_2",
                group_dimension="user_B",
                metric_name="num_calls",
                metric_value=32.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_dimension="user_D",
                metric_name="num_calls",
                metric_value=17.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_dimension="user_C",
                metric_name="num_calls",
                metric_value=18.0,
                threshold_value=50.0,
            ),
            Row(
                group="group_1",
                group_dimension="user_D",
                metric_name="num_calls_with_status_code_429",
                metric_value=17.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_dimension="user_B",
                metric_name="num_calls_with_status_code_429",
                metric_value=11.0,
                threshold_value=None,
            ),
            Row(
                group="group_1",
                group_dimension="user_B",
                metric_name="num_calls_with_status_code_429",
                metric_value=18.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_dimension="user_D",
                metric_name="num_calls_with_status_code_429",
                metric_value=11.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_dimension="user_C",
                metric_name="num_calls_with_status_code_429",
                metric_value=21.0,
                threshold_value=None,
            ),
            Row(
                group="group_2",
                group_dimension="user_A",
                metric_name="num_calls_with_status_code_429",
                metric_value=20.0,
                threshold_value=None,
            ),
        ]

        metric_output_builder = MetricOutputBuilder(mock_runmetric_client, monitor_name, signal_name, signal_metrics)
        metrics_dict = metric_output_builder.get_metrics_dict()

        assert metrics_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "groups": {
                            "user_A": {
                                "value": 18.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_B": {
                                "value": 18.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_C": {
                                "value": 18.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_D": {
                                "value": 17.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                        },
                    },
                    "group_2": {
                        "groups": {
                            "user_A": {
                                "value": 32.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_B": {
                                "value": 32.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_C": {
                                "value": 32.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_D": {
                                "value": 33.0,
                                "threshold": 50.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
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
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_D": {
                                "value": 17.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                        },
                    },
                    "group_2": {
                        "groups": {
                            "user_A": {
                                "value": 20.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_B": {
                                "value": 11.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_C": {
                                "value": 21.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                            "user_D": {
                                "value": 11.0,
                                "timeseries": {
                                    "runId": mock_runmetric_client.run_id,
                                    "metricNames": {
                                        "value": "value",
                                        "threshold": "threshold"
                                    }
                                }
                            },
                        },
                    },
                },
            },
        }
