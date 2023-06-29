# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""MCQ."""
from constants import GENERAL_COT_PROMPTS, TaskTypes, Activities
from tasks.base import Task, TaskResults
from construct_prompts import ConstructPrompt
from utils import get_predictions, make_and_log_exception, CUSTOM_DIMENSIONS
from logging_utilities import log_info
from error_definitions import OpenAIModuleError, AutoPromptInternalError, ComputeMetricsInternalError
from exceptions import AutoPromptException, ComputeMetricsException
from azureml.metrics import compute_metrics
from azureml.rag.utils.logging import get_logger, track_activity

logger = get_logger("multiple_choice")


class MCQ(Task):
    """Arithmetic QA class."""

    def __init__(self, meta_prompts=[], num_choices=None):
        """Initialize the class."""
        super().__init__()
        if meta_prompts is not None and len(meta_prompts) > 0:
            self.meta_prompts = meta_prompts
        else:
            self.meta_prompts = GENERAL_COT_PROMPTS
        self.prompt_constructor = ConstructPrompt(TaskTypes.mcq).construct_prompt
        self.best_prompt = None
        max_choice = chr(ord('A') + num_choices - 1)
        self.phase_2_prompt = "\nTherefore, among A through "+max_choice+", the answer is"

    def find_best_prompt(self, X, y, valid_x, valid_y, text_keys, **kwargs):
        """Find Best Prompt."""
        with track_activity(logger, Activities.MCQ, custom_dimensions=CUSTOM_DIMENSIONS) as mcq_activity:
            log_info(logger, "Running Phase 1", CUSTOM_DIMENSIONS)
            prompts_dict = []
            all_preds = {
                "prompt": [],
                "input": [],
                "predictions": []
            }
            with track_activity(
                logger,
                Activities.EVALUATE_PROMPTS,
                custom_dimensions=CUSTOM_DIMENSIONS
            ) as evaluate_prompts_activity:
                for base_prompt in self.meta_prompts:
                    zs_texts = X[text_keys[0]].apply(lambda x: self.prompt_constructor(x, base_prompt)).tolist()
                    log_info(logger, "Sample Prompt", CUSTOM_DIMENSIONS)
                    log_info(logger, zs_texts[0], CUSTOM_DIMENSIONS)
                    try:
                        y_pred_p1 = get_predictions(zs_texts, max_tokens=1, batch_size=2048, **kwargs)
                    except Exception as e:
                        make_and_log_exception(
                            error_cls=OpenAIModuleError,
                            exception_cls=AutoPromptException,
                            logger=logger,
                            activity_logger=evaluate_prompts_activity,
                            inner_exception=e
                        )
                    p2_texts = []
                    for base, pred in zip(zs_texts, y_pred_p1):
                        p2_prompt = base
                        p2_prompt += "\n"+pred
                        p2_prompt += self.phase_2_prompt
                        p2_texts.append(p2_prompt)

                    try:
                        y_pred = get_predictions(p2_texts, max_tokens=1, batch_size=2048, **kwargs)
                    except Exception as e:
                        make_and_log_exception(
                            error_cls=OpenAIModuleError,
                            exception_cls=AutoPromptException,
                            logger=logger,
                            activity_logger=evaluate_prompts_activity,
                            inner_exception=e
                        )
                    try:
                        metrics = compute_metrics(task_type="qa",
                                                  y_test=y.apply(lambda x: x.lower()).tolist(),
                                                  y_pred=y_pred, ignore_case=True)
                    except Exception as e:
                        make_and_log_exception(
                            error_cls=ComputeMetricsInternalError,
                            exception_cls=ComputeMetricsException,
                            logger=logger,
                            inner_exception=e,
                            activity_logger=evaluate_prompts_activity,
                            message_params={"error": repr(e)}
                        )
                    acc = metrics["metrics"]["exact_match"]
                    f1 = metrics["metrics"]["f1_score"]
                    cur_dict = {
                        "prompt": base_prompt,
                        "f1": f1,
                        "exact_match": acc,
                        "sample_prompt": p2_texts[0]
                    }
                    prompts_dict.append(cur_dict)
                    all_preds["input"] += p2_texts
                    all_preds["predictions"] += y_pred
                    all_preds["prompt"] += [base_prompt]*len(p2_texts)

            prompts_dict.sort(key=lambda x: (x["f1"], x["exact_match"]), reverse=True)
            self.best_prompt = prompts_dict[0]["prompt"]

            with track_activity(
                logger,
                Activities.GENERATE_PREDICTIONS,
                custom_dimensions=CUSTOM_DIMENSIONS
            ) as predict_activity_logger:
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
            try:
                metrics = compute_metrics(task_type="qa",
                                          y_test=valid_y.apply(lambda x: x.lower()).tolist(),
                                          y_pred=valid_preds,
                                          ignore_case=True)
            except Exception as e:
                make_and_log_exception(
                    error_cls=ComputeMetricsInternalError,
                    exception_cls=ComputeMetricsException,
                    logger=logger,
                    inner_exception=e,
                    activity_logger=mcq_activity,
                    message_params={"error": repr(e)}
                )
            val_acc = metrics["metrics"]["exact_match"]
            val_f1 = metrics["metrics"]["f1_score"]

            each_sample_metrics = {
                "exact_match": [],
                "f1_score": []
            }
            y_test = valid_y.apply(lambda x: x.lower()).tolist()
            for gt, pred in zip(y_test, valid_preds):
                try:
                    metrics = compute_metrics(task_type="qa", y_test=[gt], y_pred=[pred], ignore_case=True)
                except Exception as e:
                    make_and_log_exception(
                        error_cls=ComputeMetricsInternalError,
                        exception_cls=ComputeMetricsException,
                        logger=logger,
                        inner_exception=e,
                        activity_logger=mcq_activity,
                        message_params={"error": repr(e)}
                    )
                each_sample_metrics["exact_match"].append(metrics["metrics"]["exact_match"])
                each_sample_metrics["f1_score"].append(metrics["metrics"]["f1_score"])
            results = TaskResults(
                prompt_results=prompts_dict,
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
        y_pred_p1 = get_predictions(zs_texts, max_tokens=1, batch_size=2048, **kwargs)
        p2_texts = []
        for base, pred in zip(zs_texts, y_pred_p1):
            p2_prompt = base
            p2_prompt += "\n"+pred
            p2_prompt += self.phase_2_prompt
            p2_texts.append(p2_prompt)

        y_pred = get_predictions(p2_texts, **kwargs)
        return y_pred, p2_texts
