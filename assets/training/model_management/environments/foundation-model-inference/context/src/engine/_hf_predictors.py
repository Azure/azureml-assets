# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
""" Predictors for different tasks."""
import importlib
import types
import pandas as pd
import torch
import torch.cuda
import numpy as np
import scipy

from transformers import pipeline, Conversation
from transformers.pipelines.base import Pipeline
from abc import ABC, abstractmethod
from typing import Any, Union, List
from torch import nn
from collections import UserDict

from logging_config import configure_logger
_logger = configure_logger(__name__)


class DataLiterals:
    NER_IGNORE_TOKENS = ["", " ", "\n"]


class ModelConstants:
    MAX_SEQ_Length = 512


class Constants:
    MULTILABEL = "multi_label_classification"
    HF_SCRIPT_BASED_PREDICTION = "script_prediction"
    EXPERIMENTAL_FLAG = "exp"
    CUSTOM_CONFIG_MODULE = "custom_config_module"
    HF_CONFIG_CLASS = "hf_config_class"
    CUSTOM_TOKENIZER_MODULE = "custom_tokenizer_module"
    HF_TOKENIZER_CLASS = "hf_tokenizer_class"
    CUSTOM_MODEL_MODULE = "custom_model_module"
    HF_PRETRAINED_CLASS = "hf_pretrained_class"


class PreProcessingConstants:
    TEXT_PAIR = "text_pair"
    CONCAT = "concat"


class AdditionalPackages:
    EVALUATE_PACKAGE = "azureml-evaluate-mlflow"


class TaskTypes:
    MULTICLASS = "multiclass"
    MULTILABEL = "multilabel"
    TEXT_CLASSIFICATION = "text-classification"
    NER = "ner"
    TOKEN_CLASSIFICATION = "token-classification"
    QUESTION_ANSWERING = "question-answering"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    FILL_MASK = "fill-mask"
    TEXT_GENERATION = "text-generation"
    AUTOMATIC_SPEECH_RECOGNITION = "automatic-speech-recognition"
    TEXT_TO_IMAGE = "text-to-image"
    CONVERSATIONAL = "conversational"
    CHAT_COMPLETION = "chat-completion"


def _ensure_tensor_on_device(inputs, device):
    if isinstance(inputs, dict):
        return {name: _ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()}
    elif isinstance(inputs, UserDict):
        return UserDict({name: _ensure_tensor_on_device(tensor, device) for name, tensor in inputs.items()})
    elif isinstance(inputs, list):
        return [_ensure_tensor_on_device(item, device) for item in inputs]
    elif isinstance(inputs, tuple):
        return tuple([_ensure_tensor_on_device(item, device) for item in inputs])
    elif isinstance(inputs, torch.Tensor):
        if device == torch.device("cpu") and inputs.dtype in {torch.float16, torch.bfloat16}:
            inputs = inputs.float()
        return inputs.to(device)
    else:
        return inputs


def concat_data_columns(data, seperator):
    """
    Concatenating data
    Todo: Add more datatypes and handle series
    :param data: Incoming data to be processed
    :type data:  DF/Numpy
    :param seperator: separator to concat
    :type seperator: str
    :return: Processed data
    :rtype: list
    """
    if isinstance(data, pd.DataFrame):
        data = data.apply(lambda x: x.astype(str).str.cat(sep=seperator), axis=1).to_list()
    elif isinstance(data, pd.Series):
        data = data.to_list()
    elif isinstance(data, np.ndarray):
        data = list(map(lambda x: seperator.join(x), data))
    else:
        raise TypeError("Datatype not supported")
    return data


def process_text_pairs(data, keys):
    """
    Preprocess Text Pairs
    """
    if isinstance(data, pd.DataFrame):
        if len(keys) != 2:
            _logger.warning("Number of columns should be two. Using default processor")
            return None
        data.rename(columns={keys[0]: 'text', keys[1]: 'text_pair'}, inplace=True)
        data = data.to_dict(orient='records')
    elif isinstance(data, np.ndarray):
        if data.ndim != 2 or data.shape[1] != 2:
            _logger.warning("Array dimension not of required size. Using default processor")
            return None
        data = [{'text': val[0], 'text_pair': val[1]} for val in data]
    else:
        _logger.warning("Datatype not supported by TextProcessor. Using default processor")
        return None
    return data


def sanitize_load_args(items):
    for item in items:
        if isinstance(items[item], str) and items[item].startswith("torch."):
            items[item] = eval(items[item])
    return items


def get_custom_pipeline(**kwargs):
    custom_pipeline_module = kwargs.get("custom_pipeline_module", None)
    custom_pipeline_class = kwargs.get("custom_pipeline_class", None)
    if custom_pipeline_module is not None:
        try:
            imported_module = importlib.import_module(custom_pipeline_module)
            return getattr(imported_module, custom_pipeline_class, None)
        except ImportError as e:
            _logger.warning(f"custom_pipeline_module script not found: {e.msg}")
        except Exception as e:
            _logger.error(f"Error while loading custom pipeline: {repr(e)}")
    return None


def get_pipeline_parameters(**kwargs):
    """
    Extract relevant kwargs to set in pipeline
    @param kwargs:
    @return:
    """
    pipeline_init_args = kwargs.pop("pipeline_init_args", {})
    parameters = {"device": kwargs.get("device", None),
                  "batch_size": kwargs.get("batch_size", None),
                  "model_kwargs": sanitize_load_args(
                      {**kwargs.pop("model_hf_load_kwargs", {}), **kwargs.pop("model_kwargs", {})}),
                  "pipeline_class": get_custom_pipeline(**kwargs),
                  "trust_remote_code": kwargs.pop("trust_remote_code", True),
                  **sanitize_load_args(pipeline_init_args)}
    return parameters


def get_task_type_for_pipeline(task_type):
    if task_type in [TaskTypes.MULTILABEL, TaskTypes.MULTICLASS]:
        return TaskTypes.TEXT_CLASSIFICATION
    return task_type

def get_predictor(task_type: str, problem_type: str):
    """
    Helper function to return Predictor class based on task_type
    @param task_type: HuggingFace task type
    @param problem_type: multilabel or not
    @return: Return BasePredictor
    """
    if (task_type == "text-classification" and problem_type == Constants.MULTILABEL) or \
            task_type == TaskTypes.MULTILABEL:
        return MultilabelPredictor
    if task_type == TaskTypes.MULTICLASS or task_type == TaskTypes.TEXT_CLASSIFICATION:
        return ClassificationPredictor
    elif task_type in [TaskTypes.NER, TaskTypes.TOKEN_CLASSIFICATION]:
        return NERPredictor
    elif task_type == TaskTypes.QUESTION_ANSWERING:
        return QnAPredictor
    elif task_type == TaskTypes.SUMMARIZATION:
        return SummarizationPredictor
    elif task_type == TaskTypes.TRANSLATION or (isinstance(task_type, str) and "translation_" in task_type):
        return TranslationPredictor
    elif task_type == TaskTypes.FILL_MASK:
        return FillMaskPredictor
    elif task_type == TaskTypes.TEXT_GENERATION:
        return TextGenerationPredictor
    elif task_type == TaskTypes.CONVERSATIONAL or task_type == TaskTypes.CHAT_COMPLETION:
        return ChatCompletionPredictor
    _logger.error(f"Inference not support for task={task_type}. Please refer documentation for list of valid tasks")
    raise Exception(f"task={task_type} not supported")


class BasePredictor(ABC):
    """Abstract BasePredictor Class"""

    def __init__(self, task_type: str, model, tokenizer, config, pipeline=None):
        """
        Initialize the paramteres required by Predictor
        @param task_type: Task type to be solved
        @param model: HF model to be predicted on
        @param tokenizer: HF Tokenizer to be used for prediction
        @param config: HF Config used by model
        """
        self.task_type = task_type
        self.experimental = pipeline is not None
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.pipeline = pipeline

    @abstractmethod
    def predict(self, data: Any, **kwargs: Any):
        """
        Abstract method required to be implemented by child classed
        @param data: Any
        @param kwargs: Any
        @return:
        """
        pass

    def _parse_data(self, data, **kwargs):
        """
        Helper to parse Data
        @param data: Numpy.NDarray, DataFrame
        @return: List data
        """
        preprocess_type = kwargs.get("type", None)
        keys = kwargs.get("keys", [])
        if isinstance(data, pd.DataFrame):
            keys = data.columns.to_list() if len(keys) == 0 else keys
            data = data[keys]
        if preprocess_type == PreProcessingConstants.TEXT_PAIR:
            processed_data = process_text_pairs(data, keys)
            if processed_data is not None:
                return processed_data
        seperator = kwargs.get("sep", ". ")
        return concat_data_columns(data, seperator)

    def _get_set_label_mapping(self, **kwargs):
        """
        Get labels from model config or kwargs
        @param kwargs: dict
        @return: Train_labels
        """
        if isinstance(self.model, types.FunctionType):
            _logger.warning("model.config.id2label will not be "
                            "reset using train_label_list. This needs to be updated in the model directly. ")
            return {}
        if "train_label_list" not in kwargs and "class_labels" in kwargs:
            kwargs["train_label_list"] = kwargs["class_labels"]
        elif "train_label_list" not in kwargs and "train_labels" in kwargs:
            kwargs["train_label_list"] = kwargs["train_labels"]
        if "train_label_list" not in kwargs and not hasattr(self.model.config, 'id2label'):
            raise Exception("Either set id2label in model.config or pass train_label_list in misc_conf "
                            "while logging the model to use the .predict method")
        if "train_label_list" not in kwargs:
            _logger.warning("train_label_list has not been passed. This might cause result in incorrect labels if the"
                            " model was finetuned on a new dataset")
            train_labels = self.model.config.id2label
        else:
            train_labels = {i: value for i, value in enumerate(kwargs["train_label_list"])}
            self.model.config.id2label = train_labels
        return train_labels

    def _parse_pipeline_results(self, data, key, depth=0):
        """
        Parse output returned by pipeline
        @param data: dict/list of dict/list of list of dict
        @param key: the key to return from final dict
        @return: list/value
        """
        result = []
        if isinstance(data, dict):
            return data[key] if depth != 0 else [
                data[key]]  # Some tasks (Qna) return a dict for input array of length 1
        if isinstance(data, list):
            for data_point in data:
                result.append(self._parse_pipeline_results(data_point, key, depth + 1))
        return result

    def _format_results(self, data, result):
        """
        Output format should be same as input data format. This has been added inline with pytorch format
        @param data: Incoming data
        @param result: output of predict
        @return: formatted output
        """
        if isinstance(data, pd.DataFrame):
            try:
                predicted = pd.DataFrame(result)
                predicted.index = data.index
            except Exception:
                _logger.warning("The output returned cannot be formatted as a DataFrame object. Returning the "
                                "results as array")
                predicted = result
        else:
            predicted = result
        return predicted

    def _get_pipeline_parameters(self, **kwargs):
        """
        Extract relevant kwargs to set in pipeline
        @param kwargs:
        @return:
        """
        pipeline_init_args = kwargs.pop("pipeline_init_args", {})
        parameters = {"device": kwargs.get("device", None),
                      "batch_size": kwargs.get("batch_size", None),
                      "model_kwargs": kwargs.pop("model_kwargs", None),
                      "pipeline_class": get_custom_pipeline(**kwargs),
                      "trust_remote_code": kwargs.pop("trust_remote_code", True),
                      **sanitize_load_args(pipeline_init_args)}
        return parameters

    def _get_tokenizer_config(self, **kwargs):
        """
        Extract tokenizer config from kwargs
        :param kwargs: dict
        :return: Tokenizer Kwargs dict
        """
        output_config = {}
        if "tokenizer_config" in kwargs and isinstance(kwargs["tokenizer_config"], dict):
            output_config = kwargs["tokenizer_config"]
        if "generator_config" in kwargs and isinstance(kwargs["generator_config"], dict):
            output_config = {**output_config, **kwargs["generator_config"]}

        # Assumption is that we will never set the below parameters in tokenizer_config/generator_config
        if "addn_args" in kwargs:
            output_config = {**output_config, **kwargs["addn_args"]}
        if "max_gen_len" in output_config:
            output_config["max_new_tokens"] = output_config.pop("max_gen_len")
        return output_config

    def _preprocess_with_script(self, data, **kwargs):
        try:
            preprocess_script = kwargs.get("hf_preprocess_script", "preprocess")
            script = importlib.import_module(preprocess_script)
            return script.preprocess(data)
        except ImportError as e:
            _logger.warning(f"preprocess script not found: {e.msg}")
        except Exception as e:
            _logger.error(f"Error while processing the data using preprocess script: {repr(e)}")
        return data

    def _preprocess(self, data, **kwargs):
        data = self._preprocess_with_script(data, **kwargs)
        preprocessing_config = kwargs.get("preprocessing_config", {})
        preprocessing_config = preprocessing_config if isinstance(preprocessing_config, dict) else {}
        data = self._parse_data(data, **preprocessing_config)
        parameters = self._get_pipeline_parameters(**kwargs)
        return data, parameters



class ClassificationPredictor(BasePredictor):
    """Predictor for multi-class classification"""

    def _preprocess(self, data, **kwargs):
        """
        ToDo: USe Base class predictor after removing default processor of text_processor
        """
        data = self._preprocess_with_script(data, **kwargs)
        preprocessing_config = kwargs.get("preprocessing_config", {'type': PreProcessingConstants.TEXT_PAIR})
        preprocessing_config = preprocessing_config if isinstance(preprocessing_config, dict) else {}
        data = self._parse_data(data, **preprocessing_config)
        parameters = self._get_pipeline_parameters(**kwargs)
        return data, parameters

    def _get_predictions(self, data: Any, top_k: int = None, **kwargs: Any):
        """
        Helper function to get Predictions from model
        @param data: Dataframe, np.ndArray
        @param kwargs: Any
        @return: probas, predicted_labels
        """
        data, pipeline_kwargs = self._preprocess(data, **kwargs)
        if self.experimental:
            model_pipeline = self.pipeline
        else:
            model_pipeline = pipeline("text-classification", model=self.model, tokenizer=self.tokenizer,
                                      **pipeline_kwargs)
        tokenizer_kwargs = {'padding': "max_length", 'truncation': True, **self._get_tokenizer_config(**kwargs)}
        if top_k is not None:
            tokenizer_kwargs['top_k'] = top_k
        else:
            tokenizer_kwargs['return_all_scores'] = True
        if "max_seq_length" in kwargs and "max_length" not in tokenizer_kwargs:
            tokenizer_kwargs["max_length"] = kwargs["max_seq_length"]
        result = model_pipeline(data, **tokenizer_kwargs)
        probs = self._parse_pipeline_results(result, 'score')
        labels = self._parse_pipeline_results(result, 'label')
        return probs, labels

    def predict(self, data: Any, **kwargs: Any):
        """
        Predict method to return predicted labels
        @param data: Dataframe, np.ndArray
        @param kwargs: Any
        @return: Labels predicted by model
        """
        if not self.experimental:
            _ = self._get_set_label_mapping(**kwargs)
        _, labels = self._get_predictions(data, top_k=1, **kwargs)
        labels = [label[0] for label in labels]
        return self._format_results(data, labels)

    def predict_proba(self, data, **kwargs):
        """
        Predict method to return predicted Probabilities
        @param data: Dataframe, np.ndArray
        @param kwargs: Any
        @return: Labels predicted by model
        """
        preds, _ = self._get_predictions(data, **kwargs)
        return self._format_results(data, preds)


class MultilabelPredictor(ClassificationPredictor):
    """Predictor for multi-label classification"""

    def _get_threshold(self, **kwargs):
        """
        Get threshold from kwargs
        """
        if "threshold" in kwargs:
            return kwargs["threshold"]
        return 0.5

    def _threshold_predict(self, preds, labels, threshold=0.5):
        """
        Thresholds the probabilities and return labels
        """
        final_labels = []
        for pred, label in zip(preds, labels):
            final_labels.append(str([label[j] for j, pred_value in enumerate(pred) if pred_value >= threshold]))
        return final_labels

    def predict(self, data, **kwargs):
        """
        Predict method to return predicted labels
        @param data: Dataframe, np.ndArray
        @param kwargs: Any
        @return: Labels predicted by model
        """
        if not self.experimental:
            _ = self._get_set_label_mapping(**kwargs)
        probas, labels = self._get_predictions(data, **kwargs)
        final_labels = self._threshold_predict(probas, labels, threshold=self._get_threshold(**kwargs))
        return self._format_results(data, final_labels)

    def predict_proba(self, data, **kwargs):
        """
        Predict method to return predicted Probabilities
        @param data: Dataframe, np.ndArray
        @param kwargs: Any
        @return: Labels predicted by model
        """
        preds, _ = self._get_predictions(data, **kwargs)
        return self._format_results(data, preds)


class TokenClassificationCustomPipeline(Pipeline):
    """
    Named Entity Recognition pipeline using any `ModelForTokenClassification`. See the [named entity recognition
    examples](../task_summary#named-entity-recognition) for more information.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.train_labels = kwargs.pop("train_labels")
        self.max_seq_length = kwargs.pop("max_seq_length")
        self.label_map = {self.train_labels[key]: key for key in self.train_labels}

    def _sanitize_parameters(
            self,
            **tokenizer_config
    ):
        preprocess_params = tokenizer_config
        return preprocess_params, {}, {}

    def __call__(self, inputs: Union[str, List[str]], **kwargs):
        """
        Classify each token of the text(s) given as inputs.
        """
        self._tokenizer_config = kwargs.copy()
        return super().__call__(inputs, **kwargs)

    def preprocess(self, sentence, **preprocess_params):
        tokens = sentence.split(" ")
        # append label which will be used to align predictions only
        words = [item for item in tokens if item not in DataLiterals.NER_IGNORE_TOKENS]
        labels = ["O"] * len(words)
        tokenizer_config = {
            'max_length': self.max_seq_length,
            'padding': 'max_length',
            'truncation': True,
            'return_tensors': "pt",
            'is_split_into_words': True,
            **self._tokenizer_config
        }
        tokenized = self.tokenizer(words,
                                   None,
                                   **tokenizer_config)
        pad_id = nn.CrossEntropyLoss().ignore_index
        label_ids = np.full(self.max_seq_length, fill_value=pad_id, dtype=np.int32)

        token_idx = 1  # start with index 1 because 0 is a special token
        for label_idx in range(len(words)):
            if token_idx < self.max_seq_length:
                # set label at the starting index of the token
                label_ids[token_idx] = self.label_map[labels[label_idx]]
            token_idx += len(self.tokenizer.tokenize(words[label_idx]))
            # TODO: Remove extra tokenization step if possible ^

        # this should only be added during Split.test once we stop return labels for test split
        tokenized["labels"] = torch.LongTensor([[int(item) for item in label_ids]])
        return tokenized

    def _forward(self, model_inputs):
        # Forward
        labels = model_inputs.pop("labels")
        if self.framework == "tf":
            logits = self.model(**model_inputs)[0]
        else:
            output = self.model(**model_inputs)
            logits = output["logits"] if isinstance(output, dict) else output[0]
        return {
            "logits": logits,
            "labels": labels,
            **model_inputs
        }

    def postprocess(self, all_outputs):
        predictions, label_ids = all_outputs["logits"].detach().numpy(), all_outputs["labels"]
        preds_list, _, _ = self._align_predictions_with_proba(np.array(predictions), np.array(label_ids))
        preds_list = list(map(lambda x: str(x), preds_list))
        return preds_list

    def _align_predictions_with_proba(self, predictions, label_ids):
        """
        Helper function to align predictions with words
        @param predictions:
        @param label_ids:
        @return:
        """
        preds = np.argmax(predictions, axis=2)
        probas = scipy.special.softmax(predictions, axis=2)
        pred_probas = np.amax(probas, axis=2)
        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]
        preds_proba_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(self.train_labels[label_ids[i][j]])
                    preds_list[i].append(self.train_labels[preds[i][j]])
                    preds_proba_list[i].append(pred_probas[i][j])
        return preds_list, out_label_list, preds_proba_list


class NERPredictor(BasePredictor):
    """ Predictor for Token classification"""

    def __init__(self, task_type: str, model, tokenizer, config, pipeline=None):
        super().__init__(task_type, model, tokenizer, config, pipeline)
        if self.experimental:
            _logger.warning("Experimental not supported for NER tasks as of now")
            self.experimental = False
            self.model = model() if isinstance(self.model, types.FunctionType) else model

    def _get_pipeline_parameters(self, **kwargs):
        """
        Extract relevant kwargs to set in pipeline
        @param kwargs:
        @return:
        """
        pipeline_init_args = kwargs.pop("pipeline_init_args", {})
        parameters = {
            "model_kwargs": kwargs.pop("model_kwargs", None),
            "trust_remote_code": kwargs.pop("trust_remote_code", True),
            **sanitize_load_args(pipeline_init_args)}
        return parameters

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return predicted entity for each word
        @param data_orig: Dataframe, np.ndArray
        @param kwargs: Any
        @return: Labels predicted by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        train_labels = self._get_set_label_mapping(**kwargs)
        if "max_seq_length" not in kwargs:
            pipeline_kwargs["max_seq_length"] = min(ModelConstants.MAX_SEQ_Length, self.tokenizer.model_max_length)
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        if "max_length" in tokenizer_config:
            pipeline_kwargs["max_seq_length"] = tokenizer_config["max_length"]
        pipeline_kwargs["train_labels"] = train_labels
        model_pipeline = TokenClassificationCustomPipeline(task="token-classification", model=self.model,
                                                           tokenizer=self.tokenizer,
                                                           **pipeline_kwargs)
        result = model_pipeline(data, **tokenizer_config)
        return self._format_results(data_orig, result)


class QnAPredictor(BasePredictor):

    # Todo: Make it static method and share with QnAPipelinePredictor
    def _parse_data(self, data, **kwargs):
        """
        Helper to parse Data
        @param data: Numpy.NDarray, DataFrame
        @return: context, question
        """
        keys = kwargs.get("keys", None)
        if keys is not None and len(keys) < 2:
            raise TypeError(
                "Extractive QnA must have at least two columns set as keys in preprocessing_config corresponding to "
                "'context' and 'question'")
        if isinstance(data, pd.DataFrame):
            columns = data.columns.to_list()
            if keys is None and "context" in columns and "question" in columns:
                keys = ["context", "question"]
            elif keys is None:
                _logger.warning("Assuming first column to be context and the second column to be question")
                keys = columns
            if len(columns) < 2 or keys[0] not in columns or keys[1] not in columns:
                raise TypeError("Extractive QnA must have at least two columns in Dataframe. The columns must match "
                                "the preprocessing_config keys if set")
            question = data[keys[1]].to_list()
            context = data[keys[0]].to_list()
        elif isinstance(data, pd.Series):
            raise TypeError("For Extractive QnA at least two columns are required")
        elif isinstance(data, np.ndarray):
            if data.ndim != 2:
                raise TypeError("The second dimension must be of size 2 corresponding to 'context' and 'question'")
            context, question = np.hsplit(data, 2)
            context, question = list(context.ravel()), list(question.ravel())
        else:
            raise TypeError("Datatype not supported")
        return context, question

    def predict(self, data, **kwargs):
        """
        Predict method to return predicted answers from context for given question
        @param data: Dataframe, np.ndArray
        @param kwargs: Any
        @return: Answers identified by model
        """
        context, question = self._parse_data(data)
        pipeline_kwargs = self._get_pipeline_parameters(**kwargs)
        if not self.experimental:
            predictor = pipeline(task=self.task_type, model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        outputs = predictor(context=context, question=question, **tokenizer_config)
        result = self._parse_pipeline_results(outputs, 'answer')
        return self._format_results(data, result)


class TranslationPredictor(BasePredictor):

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return predicted answers from context for given question
        @param data: Dataframe, np.ndArray
        @param kwargs: dict
        @return: List Answers identified by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        # ToDo check for correct task_type
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        if not self.experimental:
            predictor = pipeline(task=self.task_type, model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        outputs = predictor(list(data), **tokenizer_config)
        result = self._parse_pipeline_results(outputs, 'translation_text')
        return self._format_results(data_orig, result)


class SummarizationPredictor(BasePredictor):

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return Summary of text
        @param data: Dataframe, np.ndArray
        @param kwargs: dict
        @return: List Answers identified by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        if not self.experimental:
            predictor = pipeline(task=self.task_type, model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        outputs = predictor(list(data), **tokenizer_config)
        result = self._parse_pipeline_results(outputs, 'summary_text')
        return self._format_results(data_orig, result)


class FillMaskPredictor(BasePredictor):

    def _parse_pipeline_results(self, data, key, depth=0):
        """
        Parse output returned by pipeline
        Return the token with highest score only
        @param data: dict/list of dict/list of list of dict
        @param key: the key to return from final dict
        @return: list/value
        """
        result = []
        if isinstance(data, dict):
            return data[key] if depth != 0 else [
                data[key]]  # Some tasks return a dict for input array of length 1
        if isinstance(data, list):
            for data_point in data:
                return_value = self._parse_pipeline_results(data_point, key, depth + 1)
                result.append(return_value)
                # For Fill-mask we need the token with highest score only
                if not isinstance(return_value, list) and isinstance(data_point, dict):
                    return return_value
        return result

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return predicted token
        @param data: Dataframe, np.ndArray
        @param kwargs: dict
        @return: List Answers identified by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        if not self.experimental:
            predictor = pipeline(task=self.task_type, model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        outputs = predictor(list(data), **tokenizer_config)
        result = self._parse_pipeline_results(outputs, 'token_str')
        if not isinstance(result, list):  # Input Only support ndarray or Dataframe
            return self._format_results(data_orig, [result])
        if len(data) == 1 and len(result) != 1:
            return self._format_results(data_orig, [",".join(result)])
        final_result = []
        # Final result type needs to be determined
        for res in result:
            final_result.append(",".join(res) if isinstance(res, list) and isinstance(data_orig, pd.DataFrame) else
                                res)
        return self._format_results(data_orig, final_result)


class TextGenerationPredictor(BasePredictor):

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return predicted token
        @param data: Dataframe, np.ndArray
        @param kwargs: dict
        @return: List Answers identified by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        if not self.experimental:
            predictor = pipeline(task=self.task_type, model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        outputs = predictor(list(data), **tokenizer_config)
        result = self._parse_pipeline_results(outputs, 'generated_text')
        return self._format_results(data_orig, result)


class ChatCompletionPredictor(BasePredictor):

    @classmethod
    def _create_conversation(cls, conv_arr, **kwargs):
        assert isinstance(conv_arr, list), "Each data point should be a conversation array"
        B_SYS = kwargs.get("B_SYS", "<<SYS>>\n")
        E_SYS = kwargs.get("E_SYS", "\n<</SYS>>\n\n")
        assert len(conv_arr) > 0, "Conversation is empty"
        assert conv_arr[-1]["role"] == "user"
        next_turn = "system" if conv_arr[0]["role"] == "system" else "user"
        conversation = Conversation()
        for i, conv in enumerate(conv_arr):
            if conv["role"] == "system":
                assert next_turn == "system", "System prompts can only be set at the start of the conversation"
                next_turn = "user"
                conversation.add_user_input(B_SYS + conv_arr[0]["content"].strip() + E_SYS)
                conversation.mark_processed()
            if conv["role"] == "assistant":
                assert next_turn == "assistant", "Invalid Turn. Expected user input"
                next_turn = "user"
                conversation.append_response(conv["content"].strip())
            elif conv["role"] == "user":
                assert next_turn == "user", "Invalid Turn. Expected assistant input"
                next_turn = "assistant"
                conversation.add_user_input(conv["content"].strip())
                if i != len(conv_arr[0:]) - 1:
                    conversation.mark_processed()
        return conversation

    def _parse_pipeline_results(self, data, key="", depth=0):
        """
        Parse output returned by pipeline
        @param data: dict/list of dict/list of list of dict
        @param key: the key to return from final dict
        @return: list/value
        """
        result = []
        if not isinstance(data, list):
            raise Exception("Expected conversations as output of the conversational pipeline")
        for data_point in data:
            result.append(data_point.generated_responses[-1])
        return result

    @classmethod
    def _parse_data(cls, data, **kwargs):
        """
        Helper to parse Data
        @param data: Numpy.NDarray, DataFrame
        @return: List data
        """
        if isinstance(data, pd.DataFrame):
            data = data[data.columns[0]].tolist()
        else:
            raise Exception("Only Pandas dataframe supported as of now")
        if len(data) < 1:
            raise Exception(f"Expected at least one data point. Found {len(data)}")
        if isinstance(data[0], dict):
            return [cls._create_conversation(data, **kwargs)]
        else:
            return [cls._create_conversation(conv, **kwargs) for conv in data]

    def _preprocess(self, data, **kwargs):
        data = self._preprocess_with_script(data, **kwargs)
        preprocessing_config = kwargs.get("preprocessing_config", {})
        preprocessing_config = preprocessing_config if isinstance(preprocessing_config, dict) else {}
        data = self._parse_data(data, **preprocessing_config)
        parameters = self._get_pipeline_parameters(**kwargs)
        return data, parameters

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return predicted token
        @param data: Dataframe, np.ndArray
        @param kwargs: dict
        @return: List Answers identified by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        if not self.experimental:
            predictor = pipeline(task="conversational", model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        outputs = predictor(list(data), **tokenizer_config)
        if isinstance(outputs, Conversation):
            outputs = [outputs]
        result = self._parse_pipeline_results(outputs)
        return self._format_results(data_orig, result)


class FeatureExtractionPredictor(BasePredictor):

    def predict(self, data_orig, **kwargs):
        """
        Predict method to return predicted token
        @param data: Dataframe, np.ndArray
        @param kwargs: dict
        @return: List Answers identified by model
        """
        data, pipeline_kwargs = self._preprocess(data_orig, **kwargs)
        if not self.experimental:
            predictor = pipeline(task=self.task_type, model=self.model, tokenizer=self.tokenizer, config=self.config,
                                 **pipeline_kwargs)
        else:
            predictor = self.pipeline
        tokenizer_config = self._get_tokenizer_config(**kwargs)
        outputs = predictor(list(data), **tokenizer_config)
        return self._format_results(data_orig, outputs)
