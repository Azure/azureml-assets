from promptflow import tool
import numpy as np
from constants import Metric


@tool
def concat_results(gpt_scores: dict = None,
                   f1_score: float = None
                   ) -> dict:
    load_list = [{'name': Metric.GPTCoherence, 'score': np.nan},
                 {'name': Metric.GPTSimilarity, 'score': np.nan},
                 {'name': Metric.GPTFluency, 'score': np.nan},
                 {'name': Metric.GPTRelevance, 'score': np.nan},
                 {'name': Metric.F1Score, 'score': f1_score}
                 ]

    scalar_metrics = [Metric.F1Score]
    score_list = []
    for item in load_list:
        item_name = item["name"]
        if item_name in scalar_metrics:
            try:
                score = float(item["score"])
            except Exception:
                score = np.nan
        else:
            try:
                score = float(gpt_scores[item_name])
            except Exception:
                score = np.nan
        score_list.append({"name": item_name,
                           "score": score})

    variant_level_result = {}
    for item in score_list:
        item_name = str(item["name"])
        variant_level_result[item_name] = item["score"]
    return variant_level_result
