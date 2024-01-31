# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""The component deploys the promptflow for db copilot."""
from promptflow import tool
import uuid


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def session_python_tool(chat_history: list) -> str:
    if chat_history:
        if 'session_id' in chat_history[-1]['outputs']:
            return chat_history[-1]['outputs']['session_id']
    sid = str(uuid.uuid4())
    return sid
