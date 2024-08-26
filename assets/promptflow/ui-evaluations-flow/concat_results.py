from typing import Dict

from promptflow import tool
from constants import Metric
import numpy as np


def default_safety_result(metric_name):
    return {
        metric_name: np.nan,
        metric_name + "_score": np.nan,
        metric_name + "_reason": np.nan,
    }


def default_quality_result(metric_name: str):
    return {metric_name: np.nan}


def default_groundedness_results():
    return {
        "gpt_groundedness": np.nan,
        "gpt_groundedness_reason": np.nan,
    }


@tool
def concat_results(
    f1_score_results: Dict[str, str] = None,
    groundedness_results: Dict[str, str] = None,
    gpt_coherence_results: Dict[str, str] = None,
    gpt_fluency_results: Dict[str, str] = None,
    gpt_relevance_results: Dict[str, str] = None,
    gpt_similarity_results: Dict[str, str] = None,
    hate_unfairness_results: Dict[str, str] = None,
    self_harm_results: Dict[str, str] = None,
    sexual_results: Dict[str, str] = None,
    violence_results: Dict[str, str] = None,
) -> Dict[str, str]:
    concated_results = {}

    concated_results.update(f1_score_results or default_quality_result(Metric.F1Score))
    concated_results.update(groundedness_results or default_groundedness_results)
    concated_results.update(gpt_coherence_results or default_quality_result(Metric.GPTCoherence))
    concated_results.update(gpt_fluency_results or default_quality_result(Metric.GPTFluency))
    concated_results.update(gpt_relevance_results or default_quality_result(Metric.GPTRelevance))
    concated_results.update(gpt_similarity_results or default_quality_result(Metric.GPTSimilarity))
    concated_results.update(hate_unfairness_results or default_safety_result(Metric.HateFairness))
    concated_results.update(self_harm_results or default_safety_result(Metric.SelfHarm))
    concated_results.update(sexual_results or default_safety_result(Metric.Sexual))
    concated_results.update(violence_results or default_safety_result(Metric.Violence))
    
    return concated_results
