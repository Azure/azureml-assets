from promptflow import tool
from promptflow.evals.evaluators import SelfHarmEvaluator as evaluator

from run_safety_evaluator import run_safety_evaluator


@tool
def run_selfharm_evaluator(question, answer):
    return run_safety_evaluator(evaluator, question, answer)
