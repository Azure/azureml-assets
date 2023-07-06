# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Abstractive QA."""
from constants import GENERAL_SQUAD_PROMPTS, META_PROMPTS_PREFIX, TaskTypes, DEFAULT_FALLBACK_METRICS, Activities
from tasks.base import Task, TaskResults
from construct_prompts import ConstructPrompt
from utils import (
    CUSTOM_DIMENSIONS,
    get_predictions,
    completion_with_retry,
    make_and_log_exception
)
from logging_utilities import log_info, log_warning
from error_definitions import OpenAIModuleError, AutoPromptInternalError, ComputeMetricsInternalError
from exceptions import AutoPromptException, ComputeMetricsException
from azureml.metrics import compute_metrics
from azureml.rag.utils.logging import get_logger, track_activity

import numpy as np

logger = get_logger("abstractive")


class AbstractiveQA(Task):
    """Extractive QA class."""

    def __init__(
        self,
        meta_prompts=[],
        n_prompts=10,
        best_of=100,
        primary_metric="f1_score",
        n_completions=10,
        retries=3,
    ):
        """Initialize the class."""
        super().__init__()
        if meta_prompts is not None and len(meta_prompts) > 0:
            self.meta_prompts = meta_prompts
        else:
            self.meta_prompts = GENERAL_SQUAD_PROMPTS
        self.prompt_constructor = ConstructPrompt(TaskTypes.abstractive).construct_prompt
        self.best_prompt = None
        self.meta_prompt_prefix = META_PROMPTS_PREFIX
        if n_prompts > 128:
            logger.warn("N_prompts value shouldn't be more than 128.")
        self.n_prompts = min(n_prompts, 128)
        self.best_of = min(best_of, 128)
        self.primary_metric = primary_metric
        self.n_completions = n_completions
        self.retries = retries

    def produce_prompts(self, **kwargs):
        """Produce some 'n' desired number of prompts."""
        all_prompts = []
        unique_prompts = set([])
        unfinished_prompts = 0

        model_name = kwargs['llm_config']['model_name']
        if model_name.startswith("gpt-3.5-turbo") \
            or model_name.startswith("gpt-35-turbo") \
                or model_name.startswith("gpt-4"):
            chat = True
        else:
            chat = False

        for _ in range(self.retries):
            for prompt_text in self.meta_prompts:
                out = completion_with_retry(
                        model=model_name,
                        deployment_id=kwargs['llm_config']['deployment_name'],
                        prompt=[self.meta_prompt_prefix+prompt_text],
                        max_tokens=250,
                        temperature=1.5,
                        n=self.n_completions,
                        best_of=self.best_of
                )

                for choice in out['choices']:
                    finish_reason = choice['finish_reason']
                    if finish_reason != 'stop':
                        unfinished_prompts += 1
                        continue
                    if chat:
                        content = choice['message']['content'].strip()
                    else:
                        content = choice['text'].strip()
                    all_prompts.append(content)
                    unique_prompts.add(content)

                    if len(all_prompts) == self.n_prompts:
                        log_info(logger, 'Total number of prompts: ' + str(len(all_prompts)), CUSTOM_DIMENSIONS)
                        log_info(logger, 'Total unique prompts: ' + str(len(unique_prompts)), CUSTOM_DIMENSIONS)
                        log_info(logger, 'Total unfinished prompts: ' + str(unfinished_prompts), CUSTOM_DIMENSIONS)
                        return all_prompts

        log_info(logger, 'Total number of prompts: ' + str(len(all_prompts)), CUSTOM_DIMENSIONS)
        log_info(logger, 'Total unique prompts: ' + str(len(unique_prompts)), CUSTOM_DIMENSIONS)
        log_info(logger, 'Total unfinished prompts: ' + str(unfinished_prompts), CUSTOM_DIMENSIONS)
        return all_prompts

    def find_best_prompt(self, X, y, valid_x, valid_y, text_keys, **kwargs):
        """Find Best Prompt."""
        with track_activity(
            logger,
            Activities.ABSTRACTIVE,
            custom_dimensions=CUSTOM_DIMENSIONS
        ) as abstractive_activity:
            with track_activity(logger, Activities.GENERATE_PROMPT) as produce_prompt_activity:
                try:
                    all_prompts = self.produce_prompts(**kwargs)
                except Exception as e:
                    make_and_log_exception(
                        error_cls=OpenAIModuleError,
                        exception_cls=AutoPromptException,
                        logger=logger,
                        activity_logger=produce_prompt_activity,
                        inner_exception=e
                    )

            with track_activity(logger,
                                Activities.EVALUATE_PROMPTS,
                                custom_dimensions=CUSTOM_DIMENSIONS) as evaluate_prompts_activity:
                prompt_res = []
                all_preds = {
                    "prompt": [],
                    "input": [],
                    "predictions": [],
                    "answer": []
                }
                metrics_config = {"openai_params": kwargs.get("openai_params")}
                metrics_config["questions"] = kwargs.get("questions")
                metrics_config["contexts"] = kwargs.get("contexts")
                for idx, task_description in enumerate(all_prompts):
                    zs_texts = X[text_keys[0]].apply(lambda x: self.prompt_constructor(x, task_description)).tolist()
                    try:
                        y_pred = get_predictions(zs_texts, **kwargs)
                    except Exception as e:
                        make_and_log_exception(
                            error_cls=OpenAIModuleError,
                            exception_cls=AutoPromptException,
                            logger=logger,
                            activity_logger=evaluate_prompts_activity,
                            inner_exception=e
                        )
                    # print('y_pred: ', y_pred)
                    try:
                        metrics = compute_metrics(task_type="qa", y_test=y.tolist(),
                                                  y_pred=y_pred, ignore_case=True, **metrics_config)
                    except Exception as e:
                        make_and_log_exception(
                            error_cls=ComputeMetricsInternalError,
                            exception_cls=ComputeMetricsException,
                            logger=logger,
                            inner_exception=e,
                            activity_logger=evaluate_prompts_activity,
                            message_params={"error": repr(e)}
                        )
                    dev_bert = metrics["artifacts"]["bertscore"]

                    bert_f1_macro = np.mean(dev_bert["f1"])
                    bert_recall_macro = np.mean(dev_bert["recall"])
                    bert_precision_macro = np.mean(dev_bert["precision"])
                    acc = metrics["metrics"]["exact_match"]
                    f1 = metrics["metrics"]["f1_score"]
                    task_description = str.strip(task_description, '"')
                    prompt_dict = {
                        "prompt": task_description,
                        "sample_prompt": zs_texts[0],
                        "exact_match": acc,
                        "f1_score": f1,
                        "bert_f1": bert_f1_macro,
                        "bert_recall": bert_recall_macro,
                        "bert_precision": bert_precision_macro
                    }
                    for metric_name, score in metrics["artifacts"].items():
                        if metric_name.startswith("gpt_"):
                            if not score or not isinstance(score, list) or not isinstance(score, np.ndarray):
                                log_warning(
                                    logger,
                                    "Metrics package returned empty score for metric " + metric_name,
                                    CUSTOM_DIMENSIONS)
                            try:
                                cur_score = [int(i) for i in score]
                            except Exception as ex:
                                if metric_name.startswith("gpt_"):
                                    if (isinstance(score, list) or isinstance(score, np.ndarray)) and len(score) > 0:
                                        exception_cls_name = score[0]
                                        log_warning(logger,
                                                    "Ignoring metric: " + metric_name +
                                                    "\nComputation Failed due to: " + exception_cls_name,
                                                    CUSTOM_DIMENSIONS)
                                else:
                                    log_warning(
                                        logger,
                                        "Ignoring metric: " + metric_name + " due to error: " + repr(ex),
                                        CUSTOM_DIMENSIONS)
                                if metric_name == self.primary_metric:
                                    self.primary_metric = DEFAULT_FALLBACK_METRICS
                                continue
                            prompt_dict[metric_name] = np.mean(cur_score)
                    prompt_res.append(prompt_dict)
                    all_preds["input"] += zs_texts
                    all_preds["prompt"] += [task_description]*len(zs_texts)
                    all_preds["answer"] += y.tolist()
                    all_preds["predictions"] += y_pred

            prompt_res.sort(key=lambda x: x[self.primary_metric], reverse=True)
            self.best_prompt = prompt_res[0]["prompt"]
            with track_activity(logger,
                                Activities.GENERATE_PREDICTIONS,
                                custom_dimensions=CUSTOM_DIMENSIONS) as predict_activity_logger:
                try:
                    valid_preds, valid_texts = self.infer(valid_x, text_keys, **kwargs)
                except Exception as e:
                    make_and_log_exception(
                        error_cls=AutoPromptInternalError,
                        exception_cls=AutoPromptException,
                        logger=logger,
                        activity_logger=predict_activity_logger,
                        inner_exception=e,
                        message_params={"error": repr(e)}
                    )
            val_metrics_config = {"openai_params": kwargs.get("openai_params")}
            val_metrics_config["questions"] = kwargs.get("valid_questions")
            val_metrics_config["contexts"] = kwargs.get("valid_contexts")
            try:
                metrics = compute_metrics(task_type="qa", y_test=valid_y.apply(lambda x: x.lower()).tolist(),
                                          y_pred=valid_preds, ignore_case=True, **val_metrics_config)
            except Exception as e:
                make_and_log_exception(
                    error_cls=ComputeMetricsInternalError,
                    exception_cls=ComputeMetricsException,
                    logger=logger,
                    inner_exception=e,
                    activity_logger=abstractive_activity,
                    message_params={"error": repr(e)}
                )
            val_acc = metrics["metrics"]["exact_match"]
            val_f1 = metrics["metrics"]["f1_score"]
            val_bert = metrics["artifacts"]["bertscore"]
            each_sample_metrics = {
                "exact_match": [],
                "f1_score": [],
                "bert_f1": val_bert["f1"],
                "bert_precision": val_bert["precision"],
                "bert_recall": val_bert["recall"]
            }
            for name, value in metrics["artifacts"].items():
                if name == "bertscore":
                    continue
                each_sample_metrics[name] = value
            y_test = valid_y.apply(lambda x: x.lower()).tolist()
            for gt, pred in zip(y_test, valid_preds):
                try:
                    metrics = compute_metrics(task_type="qa", metrics=["f1_score", "exact_match"],
                                              y_test=[gt], y_pred=[pred], ignore_case=True)
                except Exception as e:
                    make_and_log_exception(
                        error_cls=ComputeMetricsInternalError,
                        exception_cls=ComputeMetricsException,
                        logger=logger,
                        inner_exception=e,
                        activity_logger=abstractive_activity,
                        message_params={"error": repr(e)}
                    )
                each_sample_metrics["exact_match"].append(metrics["metrics"]["exact_match"])
                each_sample_metrics["f1_score"].append(metrics["metrics"]["f1_score"])

            results = TaskResults(
                prompt_results=prompt_res,
                validation_predictions=valid_preds,
                validation_texts=valid_texts,
                validation_metrics=each_sample_metrics,
                dev_results=all_preds,
                validation_acc=val_acc,
                validation_f1=val_f1
            )
            return results

    def infer(self, X, text_keys, **kwargs):
        """Infer."""
        zs_texts = X[text_keys[0]].apply(lambda x: self.prompt_constructor(x, self.best_prompt)).tolist()
        y_pred = get_predictions(zs_texts, **kwargs)
        return y_pred, zs_texts
