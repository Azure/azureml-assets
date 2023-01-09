# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from responsibleai import RAIInsights
from azureml.core import Run


def compute_and_upload_rai_insights():
    my_run = Run.get_context()
    print("The current run-id is: " + my_run.id)
