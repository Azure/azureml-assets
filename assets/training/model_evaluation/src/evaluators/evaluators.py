# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Evaluator."""

import ast
import pandas as pd
import numpy as np
from abc import abstractmethod

from exceptions import (
    DataValidationException,
)
from error_definitions import (
    InvalidGroundTruthColumnNameData,
    InvalidYTestCasesColumnNameData,
    InvalidGroundTruthColumnNameCodeGen
)
# TODO: Import ForecastColumns from azureml.evaluate.mlflow, when it will be
# available.
from constants import TASK, ForecastingConfigContract, ForecastColumns, TextGenerationColumns, DataFrameParams, SubTask
from image_constants import ImageDataFrameParams, ODISLiterals
from azureml.evaluate.mlflow.models.evaluation.azureml._image_od_is_evaluator import (
    ImageOdIsEvaluator,
)
from azureml._common._error_definition.azureml_error import AzureMLError
from azureml.metrics import compute_metrics, constants
from logging_utilities import get_logger, log_traceback

logger = get_logger(name=__name__)


class EvaluatorFactory:
    """Evaluator Factory Class."""

    def __init__(self):
        """__init__."""
        self._evaluators = {
            TASK.CLASSIFICATION: ClassifierEvaluator,
            TASK.CLASSIFICATION_MULTILABEL: ClassifierMultilabelEvaluator,
            "multiclass": ClassifierEvaluator,
            "tabular-regressor": RegressorEvaluator,
            TASK.REGRESSION: RegressorEvaluator,
            TASK.NER: NerEvaluator,
            "text-ner": NerEvaluator,
            TASK.TEXT_CLASSIFICATION: TextClassifierEvaluator,
            TASK.TEXT_CLASSIFICATION_MULTILABEL: ClassifierMultilabelEvaluator,
            TASK.TRANSLATION: TranslationEvaluator,
            TASK.SUMMARIZATION: SummarizationEvaluator,
            TASK.QnA: QnAEvaluator,
            TASK.FILL_MASK: FillMaskEvaluator,
            TASK.TEXT_GENERATION: TextGenerationEvaluator,
            TASK.CHAT_COMPLETION: ChatCompletionEvaluator,
            TASK.FORECASTING: ForecastingEvaluator,
            TASK.IMAGE_CLASSIFICATION: ClassifierEvaluator,
            TASK.IMAGE_CLASSIFICATION_MULTILABEL: ClassifierMultilabelEvaluator,
            TASK.IMAGE_OBJECT_DETECTION: ImageObjectDetectionInstanceSegmentationEvaluator,
            TASK.IMAGE_INSTANCE_SEGMENTATION: ImageObjectDetectionInstanceSegmentationEvaluator
        }

    def get_evaluator(self, task_type, metrics_config=None):
        """Get evaluator.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        if metrics_config.get(TextGenerationColumns.SUBTASKKEY, "") == SubTask.CODEGENERATION:
            return CodeGenerationEvaluator(task_type, metrics_config)
        return self._evaluators[task_type](task_type, metrics_config)

    def register(self, name, obj):
        """Register evaluator.

        Args:
            name (_type_): _description_
            obj (_type_): _description_
        """
        self._evaluators[name] = obj


class Evaluator:
    """Evaluator for Compute Metrics mode."""

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        self.task_type = task_type
        self.metrics_config = metrics_config

    @abstractmethod
    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate predictions.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        pass

    def _convert_predictions(self, preds):
        """Convert predictions to np array.

        Args:
            preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(preds, pd.DataFrame) and len(preds.columns) == 1:
            return preds[preds.columns[0]].to_numpy()
        if isinstance(preds, pd.DataFrame) or isinstance(preds, pd.Series):
            return preds.to_numpy()
        if isinstance(preds, list):
            return np.array(preds)
        return preds


class ClassifierEvaluator(Evaluator):
    """Classifier Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, y_pred_proba=None, **kwargs):
        """Evaluate classification.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_pred_proba (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_pred_proba = self._convert_predictions(y_pred_proba)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.CLASSIFICATION, y_test=y_test, y_pred=y_pred,
                                  y_pred_proba=y_pred_proba, **self.metrics_config)
        return metrics


class TextClassifierEvaluator(Evaluator):
    """Text Classifier Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, y_pred_proba=None, **kwargs):
        """Evaluate classification.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_pred_proba (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_pred_proba = self._convert_predictions(y_pred_proba)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_CLASSIFICATION, y_test=y_test, y_pred=y_pred,
                                  y_pred_proba=y_pred_proba, **self.metrics_config)
        return metrics


class RegressorEvaluator(Evaluator):
    """Regressor Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate regression.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.REGRESSION, y_test=y_test, y_pred=y_pred,
                                  **self.metrics_config)
        return metrics


class ClassifierMultilabelEvaluator(Evaluator):
    """Classifier Multilabel.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def _convert_predictions(self, preds):
        """Convert predictions to np array.

        Args:
            preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        preds = super()._convert_predictions(preds)
        if hasattr(preds, "ndim") and preds.ndim == 1 and len(preds) > 0 and isinstance(preds[0], str):
            preds = np.array(list(map(lambda x: ast.literal_eval(x), preds)))
        return preds

    def evaluate(self, y_test, y_pred, y_pred_proba=None, **kwargs):
        """Evaluate multilabel.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_pred_proba (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_pred_proba = self._convert_predictions(y_pred_proba)
        y_test = self._convert_predictions(y_test)
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_CLASSIFICATION_MULTILABEL, y_test=y_test,
                                  y_pred=y_pred, y_pred_proba=y_pred_proba, **self.metrics_config)
        return metrics


class NerEvaluator(Evaluator):
    """NER Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def _convert_predictions(self, preds):
        """Convert predictions.

        Args:
            preds (_type_): _description_

        Returns:
            _type_: _description_
        """
        preds = super()._convert_predictions(preds)
        if hasattr(preds, "ndim") and preds.ndim == 1 and len(preds) > 0 and isinstance(preds[0], str):
            preds = np.array(list(map(lambda x: ast.literal_eval(x), preds)))
        return preds

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate NER.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred).tolist()
        y_test = self._convert_predictions(y_test).tolist()
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_NER, y_test=y_test, y_pred=y_pred,
                                  **self.metrics_config)
        return metrics


class SummarizationEvaluator(Evaluator):
    """Summarization Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate summarizer.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        if y_test.ndim == 1:
            y_test = np.reshape(y_test, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.SUMMARIZATION, y_test=y_test.tolist(),
                                  y_pred=y_pred.tolist(), **self.metrics_config)
        return metrics


class QnAEvaluator(Evaluator):
    """QnA Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate QnA.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred).tolist()
        y_test = self._convert_predictions(y_test).tolist()

        # y_pred = []
        # for pred in y_pred:
        #     if (isinstance(pred, list) or isinstance(pred, np.ndarray)):
        #         if len(pred) >= 1:
        #             y_pred.append(pred[0])
        #     else:
        #         y_pred.append(pred)
        # logger.warning("Multiple ground truths are not supported for question-answering task currently.\
        #                 Considering only the first ground truth in case of multiple values.")
        metrics = compute_metrics(task_type=constants.Tasks.QUESTION_ANSWERING, y_test=y_test,
                                  y_pred=y_pred, **self.metrics_config)
        return metrics


class TranslationEvaluator(Evaluator):
    """Translation Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate Translation.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        if y_test.ndim == 1:
            y_test = np.reshape(y_test, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.TRANSLATION, y_test=y_test.tolist(),
                                  y_pred=y_pred.tolist(), **self.metrics_config)
        return metrics


class FillMaskEvaluator(Evaluator):
    """Fill Mask Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate Fill Mask.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred).tolist()
        y_test = self._convert_predictions(y_test).tolist()
        metrics = compute_metrics(task_type=constants.Tasks.FILL_MASK, y_test=y_test,
                                  y_pred=y_pred, **self.metrics_config)
        return metrics


class TextGenerationEvaluator(Evaluator):
    """Text Generation Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate TextGeneration.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)
        if y_test.ndim == 1:
            y_test = np.reshape(y_test, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.TEXT_GENERATION, y_test=y_test.tolist(),
                                  y_pred=y_pred.tolist(), **self.metrics_config)
        return metrics


class CodeGenerationEvaluator(Evaluator):
    """Code Generation Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate TextGeneration.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            y_test_cases: list of strings

        Returns:
            _type_: _description_
        """
        y_test_cases = None
        extra_cols = kwargs.get(DataFrameParams.Extra_Cols, None)
        y_test_col_name = kwargs.get(DataFrameParams.Ground_Truth_Column_Name, None)
        # TodO: This validaion should ideally be done while data loading. Design needs to be reconsidered
        if extra_cols is None and y_test_col_name is None:
            exception = DataValidationException._with_error(
                AzureMLError.create(InvalidGroundTruthColumnNameCodeGen)
            )
            log_traceback(exception, logger)
            raise exception
        y_test_case_col_name = extra_cols[0] if extra_cols is not None and len(extra_cols) > 0 else None
        if y_test_case_col_name is not None:
            if y_test_case_col_name in y_test.columns:
                y_test_cases = y_test[[y_test_case_col_name]]
            else:
                exception = DataValidationException._with_error(
                    AzureMLError.create(InvalidYTestCasesColumnNameData)
                )
                log_traceback(exception, logger)
                raise exception

        if y_test_col_name is not None:
            if y_test_col_name in y_test.columns:
                y_test = y_test[[y_test_col_name]]
            else:
                exception = DataValidationException._with_error(
                    AzureMLError.create(InvalidGroundTruthColumnNameData)
                )
                log_traceback(exception, logger)
                raise exception
        else:
            y_test = None

        y_pred = self._convert_predictions(y_pred)
        if y_test is not None:
            y_test = self._convert_predictions(y_test)
            if y_test.ndim == 1:
                y_test = np.reshape(y_test, (-1, 1))
                y_test = y_test.tolist()
        if y_test_cases is not None:
            y_test_cases = self._convert_predictions(y_test_cases).tolist()
        if y_pred.ndim == 1:
            y_pred = np.reshape(y_pred, (-1, 1))
        metrics = compute_metrics(task_type=constants.Tasks.CODE_GENERATION, y_test=y_test,
                                  y_pred=y_pred.tolist(), test_cases=y_test_cases, **self.metrics_config)
        return metrics


class ChatCompletionEvaluator(Evaluator):
    """Chat Completion Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate Chat Completion.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred_formatted = y_pred.iloc[:, 0].apply(lambda x: [item['input_string'] for item in x])
        metrics = compute_metrics(task_type=constants.Tasks.CHAT_COMPLETION, y_pred=y_pred_formatted.tolist(),
                                  **self.metrics_config)
        return metrics


class ForecastingEvaluator(Evaluator):
    """Forecasting Evaluator.

    Args:
        Evaluator (_type_): _description_
    """

    def __init__(self, task_type, metrics_config):
        """__init__.

        Args:
            task_type (_type_): _description_
            metrics_config (_type_): _description_
        """
        super().__init__(task_type, metrics_config)

    def evaluate(self, y_test, y_pred, X_test, **kwargs):
        """Evaluate regression.

        Args:
            y_test (_type_): _description_
            y_pred (_type_): _description_
            X_test (_type_): _description_

        Returns:
            _type_: _description_
        """
        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)

        # In the evaluation component we do not have the model, so we cannot aggregate data.
        # In case user provided X_train or y train we will remove it from kwargs.
        kwargs.pop('X_train', None)
        kwargs.pop('y_train', None)
        # Take forecasting-specific parameters from the model.
        kwargs["time_series_id_column_names"] = self.metrics_config.get(
            ForecastingConfigContract.TIME_SERIES_ID_COLUMN_NAMES)
        kwargs["time_column_name"] = self.metrics_config.get(
            ForecastingConfigContract.TIME_COLUMN_NAME)
        metrics = compute_metrics(
            task_type=constants.Tasks.FORECASTING,
            y_test=y_test,
            y_pred=y_pred,
            X_test=X_test,
            **kwargs
        )
        X_test[ForecastColumns._ACTUAL_COLUMN_NAME] = y_test
        X_test[ForecastColumns._FORECAST_COLUMN_NAME] = y_pred
        return metrics


class ImageObjectDetectionInstanceSegmentationEvaluator(Evaluator):
    """Image object detection and instance segmentation Evaluator."""

    def __init__(self, task_type, metrics_config):
        """Initialize evaluator.

        Args:
            task_type (str): evaluator task type
            metrics_config (Dict): Dict of metrics config
        """
        super().__init__(task_type, metrics_config)
        self.masks_required = task_type == TASK.IMAGE_INSTANCE_SEGMENTATION

    def evaluate(self, y_test, y_pred, **kwargs):
        """Evaluate Object Detection/Instance Segmentation.

        Args:
            y_test (pd.DataFrame): pandas DataFrame with columns ["labels"]
            y_pred (pd.DataFrame): pandas DataFrame with columns ["predictions"]
            X_test (pd.DataFrame): Pandas DataFrame with columns ["image", "image_meta_info"].
        Returns:
            Dict: Dict of metrics

        """

        def _recast(label_or_pred_dict):
            if len(label_or_pred_dict[ODISLiterals.BOXES]) == 0:
                return label_or_pred_dict
            if isinstance(label_or_pred_dict[ODISLiterals.BOXES][0], np.ndarray):
                # When using MLTable input, the boxes are stored as a array of numpy arrays
                label_or_pred_dict[ODISLiterals.BOXES] = np.stack(label_or_pred_dict[ODISLiterals.BOXES], axis=0)
            elif isinstance(label_or_pred_dict[ODISLiterals.BOXES][0], list):
                # When using json, the boxes, classes, scores, labels are stored as a lists (of lists)
                label_or_pred_dict[ODISLiterals.BOXES] = np.array(label_or_pred_dict[ODISLiterals.BOXES])
                if ODISLiterals.SCORES in label_or_pred_dict:
                    label_or_pred_dict[ODISLiterals.SCORES] = np.array(label_or_pred_dict[ODISLiterals.SCORES])
                else:
                    label_or_pred_dict[ODISLiterals.LABELS] = np.array(label_or_pred_dict[ODISLiterals.LABELS])

            if ODISLiterals.MASKS in label_or_pred_dict and isinstance(label_or_pred_dict[ODISLiterals.MASKS],
                                                                       np.ndarray):
                label_or_pred_dict[ODISLiterals.MASKS] = list(label_or_pred_dict[ODISLiterals.MASKS])

            return label_or_pred_dict

        image_meta_info = y_test[ImageDataFrameParams.IMAGE_META_INFO]

        y_test = y_test.drop(ImageDataFrameParams.IMAGE_META_INFO, axis=1)

        # Convert predictions to expected format
        y_test[ImageDataFrameParams.LABEL_COLUMN_NAME] = \
            y_test[ImageDataFrameParams.LABEL_COLUMN_NAME].apply(lambda x: _recast(x))

        y_pred = self._convert_predictions(y_pred)
        y_test = self._convert_predictions(y_test)

        metrics = ImageOdIsEvaluator.compute_metrics(y_test=y_test,
                                                     y_pred=y_pred,
                                                     image_meta_info=image_meta_info,
                                                     masks_required=self.masks_required,
                                                     **self.metrics_config)

        return metrics
