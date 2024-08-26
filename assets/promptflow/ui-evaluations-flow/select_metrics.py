from typing import Dict

from promptflow import tool
from constants import Metric, MetricGroup
from promptflow._utils.logger_utils import logger


def select_metrics_from_metric_list(
    user_selected_metrics: str,
    supported_metrics: tuple,
):
    metric_dict = {}
    for metric in supported_metrics:
        if metric in user_selected_metrics or len(user_selected_metrics) == 0:
            metric_dict[metric] = True
        else:
            metric_dict[metric] = False
    return metric_dict


@tool
def select_metrics(metrics: str) -> Dict[str, bool]:
    supported_quality_metrics = Metric.QUALITY_METRICS
    supported_safety_metrics = Metric.CONTENT_HARM_METRICS
    supported_metrics = supported_quality_metrics | supported_safety_metrics
    metric_selection_dict = {}
    metric_selection_dict[MetricGroup.QUALITY_METRICS] = \
        select_metrics_from_metric_list(metrics.lower(),
                                        supported_quality_metrics)
    metric_selection_dict[MetricGroup.SAFETY_METRICS] = \
        select_metrics_from_metric_list(metrics.lower(),
                                        supported_safety_metrics)
    user_selected_metrics = [metric.replace("\"", "").replace("'", "").strip()
                             for metric in metrics.split(',') if metric]
    for user_metric in user_selected_metrics:
        if user_metric not in supported_metrics:
            logger.error("%s is not supported for evaluation"
                         % user_metric)
    return metric_selection_dict
