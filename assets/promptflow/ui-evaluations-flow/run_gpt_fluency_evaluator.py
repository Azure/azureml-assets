from promptflow import tool
from promptflow.evals.evaluators import FluencyEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_fluency_evaluator(connection, deployment_name, question, answer):
    inputs = {
        "question": question,
        "answer": answer,
    }
    return run_gpt_based_evaluator(
        connection,
        FluencyEvaluator,
        inputs,
        deployment_name
    )
