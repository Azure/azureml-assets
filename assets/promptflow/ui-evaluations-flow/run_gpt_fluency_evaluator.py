from promptflow import tool
from promptflow.evals.evaluators import FluencyEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_fluency_evaluator(question, answer):
    inputs = {
        "question": question,
        "answer": answer,
    }
    return run_gpt_based_evaluator(
        FluencyEvaluator,
        inputs,
    )
