# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from azureml.core import Run


def compute_and_upload_rai_insights():
    my_run = Run.get_context()
    print("The current run-id is: " + my_run.id)

    try:
        from responsibleai import RAIInsights
        print("Successfully imported from responsibleai")
    except Exception:
        print("Couldn't import from responsibleai")
