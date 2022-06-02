from asyncio import subprocess
from logging import exception
from pathlib import Path
import os
import sys
import datetime
import subprocess
from azureml.core import Workspace
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

subscription_id = sys.argv[1]
workspace_name = sys.argv[2]
resource_group_name = sys.argv[3]

DIR = Path("azureml-assets/latest/tests")

ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name)
submitted_job_list = []
for x in os.listdir(DIR.__str__()):
    print(f'Loading test job {x}')
    test_job = load_job(DIR.__str__()+"/"+x)
    print(f'Running test job {x}')
    ml_client.jobs.create_or_update(test_job)
    submitted_job_list.append(test_job)

succeeded_jobs = []
failed_jobs = []

while(submitted_job_list.count > 0):
    for job in submitted_job_list:
        returned_job = ml_client.jobs.get(job)
        if(returned_job.status == "Completed"):
            succeeded_jobs.append(returned_job.display_name)
            submitted_job_list.remove(job)
        elif(returned_job.status == "Failed"):
            failed_jobs.append(returned_job.display_name)
            submitted_job_list.remove(job)
        
print(f"Totally {succeeded_jobs.count+failed_jobs.count} jobs have been run. {succeeded_jobs.count} jobs succeeded. {failed_jobs.count} jobs failed")

if(failed_jobs.count>0):
    failed_job_str = ", ".join(failed_jobs)
    raise Exception(f"{failed_jobs.count} jobs failed. {failed_job_str}.")