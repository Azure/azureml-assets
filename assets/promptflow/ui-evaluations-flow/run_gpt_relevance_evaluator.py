from promptflow import tool
from promptflow.evals.evaluators import RelevanceEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_relevance_evaluator(answer, context, ground_truth):
    inputs = {
        "answer": answer,
        "context": context,
        "ground_truth": ground_truth,
    }
    return run_gpt_based_evaluator(
        RelevanceEvaluator,
        inputs,
    )
