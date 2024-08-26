# from ..utils import get_project_scope
from azure.identity import DefaultAzureCredential


def run_safety_evaluator(evaluator, question, answer):
    eval_fn = evaluator(
       {
           "subscription_id": "",
           "resource_group_name": "",
           "project_name": "",
       },
       DefaultAzureCredential(),
    )
    return eval_fn(
        question=question,
        answer=answer,
    )
