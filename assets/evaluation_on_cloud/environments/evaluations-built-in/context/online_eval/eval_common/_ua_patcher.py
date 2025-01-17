# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Patch User Agent in azure.ai.evaluation."""
import inspect

from azure.ai.evaluation import _user_agent


def get_eval_type():
    """Get evaluation type."""
    full_stack = inspect.stack()
    eval_type = ""
    for stack in full_stack:
        if "evaluate_on_data.py" in stack.filename:
            eval_type = "remote-evaluation"
            break
        elif "evaluate_online.py" in stack.filename:
            eval_type = "online-evaluation"
            break
    return eval_type


eval_type = get_eval_type()
_user_agent.USER_AGENT += "/{}".format(eval_type)
