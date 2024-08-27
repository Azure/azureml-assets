from promptflow import tool
from promptflow.evals.evaluators import SexualEvaluator as evaluator

from run_safety_evaluator import run_safety_evaluator


@tool
def run_sexual_evaluator(subscription_id, resource_group, project_name, question, answer):
    return run_safety_evaluator(subscription_id, resource_group, project_name, evaluator, question, answer)
