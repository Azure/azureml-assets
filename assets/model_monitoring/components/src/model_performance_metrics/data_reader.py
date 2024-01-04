# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the data reader for model performance compute metrics."""

from shared_utilities.io_utils import read_mltable_in_spark
from shared_utilities.momo_exceptions import DataNotFoundError


class DataReaderFactory:
    """Data reader factory class."""

    def __init__(self):
        """Reader factory to get the reader for the task."""
        self.default_reader = BaseTaskReader

    def get_reader(self, task):
        """
        Get reader for the task.

        Args:
            task: str

        Returns: reader object

        """
        # We would need different readers for tasks like text-generation(code)
        return self.default_reader(task)


class MetricsDTO:
    """Metrics DTO."""

    def __init__(self, ground_truth, predictions):
        """
        Metrics DTO.

        Args:
            ground_truth: Ground truth data
            predictions: Predictions data
        """
        self.ground_truth = ground_truth
        self.predictions = predictions


class BaseTaskReader:
    """Class for task reader."""

    def __init__(self, task):
        """
        Class for task reader.

        Args:
            task: str
        """
        self.task = task

    def read_data(self, ground_truths_column_name, file_name, predictions_column_name):
        """
        Read data for the task.

        Args:
            ground_truths_column_name: str
            file_name: str
            predictions_column_name: str

        Returns: MetricsDTO

        """
        df = read_mltable_in_spark(file_name)

        if df.isEmpty():
            raise DataNotFoundError("The production data and ground truth data are required.")

        ground_truth = df.select(ground_truths_column_name).toPandas()

        predictions = df.select(predictions_column_name).toPandas()

        return MetricsDTO(ground_truth, predictions)
