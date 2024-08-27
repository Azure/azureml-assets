from promptflow import tool
from promptflow.evals.evaluators import GroundednessEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_groundedness_evaluator(connection,deployment_name, answer, context):
    inputs = {
        "answer": answer,
        "context": context,
    }
    return run_gpt_based_evaluator(
        connection,
        GroundednessEvaluator,
        inputs,
        deployment_name
    )
