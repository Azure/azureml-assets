from azure.identity import DefaultAzureCredential


def run_safety_evaluator(subscription_id, resource_group, project_name, evaluator, question, answer):
    eval_fn = evaluator(
       {
           "subscription_id": subscription_id,
           "resource_group_name": resource_group,
           "project_name": project_name,
       },
       DefaultAzureCredential(),
    )
    return eval_fn(
        question=question,
        answer=answer,
    )
