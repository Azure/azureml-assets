# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""A basic entry script."""

# imports
import uuid
import os
from datetime import datetime
from azureml_inference_server_http.api.aml_response import AMLResponse
from azureml_inference_server_http.api.aml_request import rawhttp


def init():
    """Sample init function."""
    print("Initializing")


@rawhttp
def run(input_data):
    """Sample run function."""
    print('A new request received~~~')
    try:
        r = dict()
        r['request_id'] = str(uuid.uuid4())
        r['now'] = datetime.now().strftime("%Y/%m/%d %H:%M:%S %f")
        r['pid'] = os.getpid()
        r['message'] = "this is a sample"

        return AMLResponse(r, 200, json_str=True)
    except Exception as e:
        error = str(e)

        return AMLResponse({'error': error}, 500, json_str=True)
