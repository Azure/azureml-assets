# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the data reader for model performance compute metrics."""

from shared_utilities.io_utils import read_mltable_in_spark


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

    def _read_mltable_to_pd_dataframe(self, file_name, columns):
        """
        Read mltable to pandas dataframe.

        Args:
            file_name: str
            columns: list

        Returns: pd.DataFrame

        """
        df = read_mltable_in_spark(file_name)
        if columns is not None:
            df = df.select(columns)  # We might need to accept multiple columns in code-gen
        return df.toPandas()

    def read_data(self, ground_truths, ground_truths_column_name, predictions, predictions_column_name):
        """
        Read data for the task.

        Args:
            ground_truths: str
            ground_truths_column_name: str
            predictions: str
            predictions_column_name: str

        Returns: MetricsDTO

        """
        ground_truth = self._read_mltable_to_pd_dataframe(ground_truths,
                                                          [ground_truths_column_name]
                                                          if ground_truths_column_name is not None else None)
        predictions = self._read_mltable_to_pd_dataframe(predictions,
                                                         [predictions_column_name]
                                                         if predictions_column_name is not None else None)
        return MetricsDTO(ground_truth, predictions)
