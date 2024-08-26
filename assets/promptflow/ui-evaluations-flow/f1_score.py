from promptflow import tool
from promptflow.evals.evaluators import F1ScoreEvaluator


@tool
def compute_f1_score(ground_truth: str, answer: str) -> str:
    eval_fun = F1ScoreEvaluator()
    return eval_fun(answer=answer, ground_truth=ground_truth)
