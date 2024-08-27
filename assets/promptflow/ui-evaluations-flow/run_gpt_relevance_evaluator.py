from promptflow import tool
from promptflow.evals.evaluators import RelevanceEvaluator

from run_gpt_based_evaluator import run_gpt_based_evaluator


@tool
def run_gpt_relevance_evaluator(deployment_name, answer, context, question):
    inputs = {
        "answer": answer,
        "context": context,
        "question": question,
    }
    return run_gpt_based_evaluator(
        RelevanceEvaluator,
        inputs,
        deployment_name
    )
