from promptflow import tool
from promptflow.evals.evaluators import CoherenceEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_coherence_evaluator(deployment_name, question, answer):
    inputs = {
        "question": question,
        "answer": answer,
    }
    return run_gpt_based_evaluator(
        CoherenceEvaluator,
        inputs,
        deployment_name
    )
