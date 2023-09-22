from shared_utilities.io_utils import read_mltable_in_spark


class DataReaderFactory:

    def __init__(self):
        """

        """
        self.default_reader = BaseTaskReader

    def get_reader(self, task):
        """

        Args:
            task:

        Returns:

        """
        # We would need different readers for tasks like text-generation(code)
        return self.default_reader(task)


class MetricsDTO:

    def __init__(self, ground_truth, predictions):
        self.ground_truth = ground_truth
        self.predictions = predictions


class BaseTaskReader:
    """

    """

    def __init__(self, task):
        """

        Args:
            task:
        """
        self.task = task

    def _read_mltable(self, file_name, columns):
        """

        Args:
            file_name:
            columns:

        Returns:

        """
        df = read_mltable_in_spark(file_name)
        if columns is not None:
            df.select(columns)  # We might need to accept multiple columns in code-gen
        return df.toPandas()

    def read_data(self, ground_truths, ground_truths_column_name, predictions, predictions_column_name):
        """

        Args:
            ground_truths:
            ground_truths_column_name:
            predictions:
            predictions_column_name:

        Returns:

        """
        ground_truth = self._read_mltable(ground_truths, [
            ground_truths_column_name] if ground_truths_column_name is not None else None)
        predictions = self._read_mltable(predictions,
                                         [predictions_column_name] if predictions_column_name is not None else None)
        return MetricsDTO(ground_truth, predictions)
