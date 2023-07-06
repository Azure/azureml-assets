# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Pytest sample for demo."""
import azure.ai.ml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import os
from pathlib import Path
import time


def submit_pytest_job():
    """Submit a job."""
    subscription_id = os.environ.get('subscription_id')
    resource_group = os.environ.get('resource_group')
    workspace = os.environ.get('workspace')
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    parent = Path(__file__).resolve().parent
    test_job = azure.ai.ml.load_job(parent / 'pipeline.yml')
    print('Running test job pipeline.yml')
    test_job = ml_client.jobs.create_or_update(test_job)

    while True:
        returned_job = ml_client.jobs.get(test_job.name)
        if returned_job.status == "Completed":
            return True
        elif returned_job.status in ["Failed", "Cancelled"]:
            return False
        time.sleep(10)


def test_answer():
    """Function to check the job result."""
    assert submit_pytest_job()
