"""Base."""
from abc import ABC, abstractmethod
from constants import ALL_METRICS
from azureml.rag.utils.logging import get_logger


logger = get_logger("base_task")


class Task(ABC):
    """Task Base class."""

    def __init__(self):
        """Initialize the class."""
        pass

    @abstractmethod
    def find_best_prompt(self, X, y, valid_x, valid_y, text_keys, **kwargs):
        """Find Best Prompt Abstract Method."""
        pass

    @abstractmethod
    def infer(self, X, text_keys, **kwargs):
        """Infer Abstract Method."""
        pass


class TaskResults:
    """TaskResults Base class."""

    def __init__(self,
                 prompt_results,
                 validation_predictions,
                 validation_texts,
                 validation_metrics,
                 validation_acc,
                 validation_f1,
                 dev_results):
        """Initialize the class."""
        self._prompt_results = prompt_results
        self._validation_predictions = validation_predictions
        self._validation_texts = validation_texts
        self._validation_metrics = validation_metrics
        self._dev_results = dev_results
        self._validation_accuracy = validation_acc
        self._validation_f1 = validation_f1
        self.set_metric_attrs()

    def set_metric_attrs(self):
        """Set Metrics name as class attributes."""
        for metric_name in ALL_METRICS:
            value = None
            if metric_name in self._validation_metrics:
                value = self._validation_metrics[metric_name]
            else:
                logger.warning(f"{metric_name} not found in metrics results.")
                continue
            setattr(self, metric_name, value)

    @property
    def prompt_results(self):
        """Full AutoPrompt Results DataFrame."""
        return self._prompt_results

    @property
    def validation_predictions(self):
        """Validation data predictions DataFrame."""
        return self._validation_predictions

    @property
    def validation_texts(self):
        """Validation Inputs constructed using prompt."""
        return self._validation_texts

    @property
    def dev_results(self):
        """Autoprompt results on dev dataset for each datapoint."""
        return self._dev_results

    @property
    def validation_acc(self):
        """Validation Accuracy."""
        return self._validation_accuracy

    @property
    def validation_f1(self):
        """Validation F1 Score."""
        return self._validation_f1
