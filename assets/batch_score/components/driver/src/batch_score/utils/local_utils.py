# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Local mode utilities."""

from azureml.core import Run


def is_running_in_azureml_job() -> bool:
    """Check if the batch score client is being run inside an AzureML job."""
    run = Run.get_context()
    # When running locally, the run ID will start with "OfflineRun".
    # e.g. "OfflineRun_f48930fe-5d7c-4e95-8b3f-fcaab354b9a9"
    return not run.id.startswith("OfflineRun")
