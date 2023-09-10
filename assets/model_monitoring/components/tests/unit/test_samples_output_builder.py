# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains unit tests for the outputter's samples output builder."""

import pytest
from typing import List
from pyspark.sql import Row

from model_monitor_metric_outputter.builder.samples_output_builder import (
    SamplesOutputBuilder,
)


@pytest.mark.unit2
class TestSamplesOutputBuilder:
    """Test class for samples output builder."""

    def test_samples_with_1_level_groups(self):
        """Test samples output builder for samples with one level samples groups."""
        samples: List[Row] = [
            Row(
                group="group_1",
                group_dimension="",
                metric_name="num_calls",
                samples_name="My Samples",
                asset="azureml:my_samples:1",
            )
        ]

        samples_output_builder = SamplesOutputBuilder(samples)
        samples_dict = samples_output_builder.get_samples_dict()

        assert samples_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "samples": {"My Samples": {"uri": "azureml:my_samples:1"}}
                    }
                }
            }
        }

    def test_samples_with_2_level_groups(self):
        """Test samples output builder for samples with two level samples groups."""
        samples: List[Row] = [
            Row(
                group="group_1",
                group_dimension="user_A",
                metric_name="num_calls",
                samples_name="My Samples",
                asset="azureml:my_samples:1",
            )
        ]

        samples_output_builder = SamplesOutputBuilder(samples)
        samples_dict = samples_output_builder.get_samples_dict()

        assert samples_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "groups": {
                            "user_A": {
                                "samples": {
                                    "My Samples": {"uri": "azureml:my_samples:1"}
                                }
                            }
                        }
                    }
                }
            }
        }

    def test_multiple_samples_with_different_groups(self):
        """Test samples output builder for two samples with different groups."""
        samples_samples: List[Row] = [
            Row(
                group="group_1",
                group_dimension="user_A",
                metric_name="num_calls",
                samples_name="My Samples",
                asset="azureml:my_samples:1",
            ),
            Row(
                group="group_2",
                group_dimension="user_A",
                metric_name="num_calls",
                samples_name="My Samples",
                asset="azureml:my_samples:1",
            ),
        ]

        samples_output_builder = SamplesOutputBuilder(samples_samples)
        samples_dict = samples_output_builder.get_samples_dict()

        assert samples_dict == {
            "num_calls": {
                "groups": {
                    "group_1": {
                        "groups": {
                            "user_A": {
                                "samples": {
                                    "My Samples": {"uri": "azureml:my_samples:1"}
                                }
                            }
                        }
                    },
                    "group_2": {
                        "groups": {
                            "user_A": {
                                "samples": {
                                    "My Samples": {"uri": "azureml:my_samples:1"}
                                }
                            }
                        }
                    },
                }
            }
        }

    def test_multiple_samples_with_no_groups(self):
        """Test samples output builder for two samples with no groups and different metrics."""
        samples: List[Row] = [
            Row(
                group="",
                group_dimension="",
                metric_name="num_calls_1",
                samples_name="My Samples1",
                asset="azureml:my_samples1:1",
            ),
            Row(
                group="",
                group_dimension="",
                metric_name="num_calls_2",
                samples_name="My Samples2",
                asset="azureml:my_samples2:1",
            ),
        ]

        samples_output_builder = SamplesOutputBuilder(samples)
        samples_dict = samples_output_builder.get_samples_dict()

        assert samples_dict == {
            "num_calls_1": {
                "samples": {"My Samples1": {"uri": "azureml:my_samples1:1"}}
            },
            "num_calls_2": {
                "samples": {"My Samples2": {"uri": "azureml:my_samples2:1"}}
            },
        }

    def test_multiple_samples_conflicting_samples(self):
        """Test samples output builder for conflicting samples."""
        samples: List[Row] = [
            Row(
                group="group_1",
                group_dimension="user_A",
                metric_name="num_calls",
                samples_name="My Samples",
                asset="azureml:my_samples:1",
            ),
            Row(
                group="group_1",
                group_dimension="user_A",
                metric_name="num_calls",
                samples_name="My Samples",
                asset="azureml:my_samples:1",
            ),
        ]

        with pytest.raises(Exception) as e:
            SamplesOutputBuilder(samples)
        assert (
            "num_calls.groups.group_1.groups.user_A.samples.My Samples.uri"
            in e.value.args[0]
        )
