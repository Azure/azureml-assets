from promptflow import tool
from promptflow.evals.evaluators import SimilarityEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_similarity(question, answer, ground_truth):
    inputs = {
        "question": question,
        "answer": answer,
        "ground_truth": ground_truth,
    }
    return run_gpt_based_evaluator(
        SimilarityEvaluator,
        inputs,
    )
