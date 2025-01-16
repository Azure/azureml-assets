import inspect

from azure.ai.evaluation import _user_agent
from azure.ai.evaluation._version import VERSION


def get_eval_type():
    stack = inspect.stack()
    eval_type = ""
    if "evaluate_on_data.py" in stack[1].filename:
        eval_type = "remote-evaluation"
    elif "evaluate_online.py" in stack[1].filename:
        eval_type = "online-evaluation"
    return eval_type


eval_type = get_eval_type()
_user_agent.USER_AGENT = "{}/{}/{}".format("azure-ai-evaluation", VERSION, eval_type)
