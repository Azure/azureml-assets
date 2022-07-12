# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import argparse
from subprocess import check_call
from pathlib import Path
import azure.ai.ml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import yaml
import os
import sys
TEST_YML = "tests.yml"

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
    parser.add_argument("-g", "--test-group", required=True, type=str, help="test group name")
    parser.add_argument("-s", "--subscription", required=True, type=str, help="Subscription ID")
    parser.add_argument("-r", "--resource-group", required=True, type=str, help="Resource group name")
    parser.add_argument("-w", "--workspace-name", required=True, type=str, help="Workspace name")
    args = parser.parse_args()
    tests_dir = args.input_dir
    test_group = args.test_group
    subscription_id = args.subscription
    resource_group = args.resource_group
    workspace = args.workspace_name
    # default workspace info
    group_pre = None
    group_post = None

    with open(tests_dir / TEST_YML) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        if 'pre' in data[test_group]:
            group_pre = tests_dir / data[test_group]['pre']
        if 'post' in data[test_group]:
            group_post = tests_dir / data[test_group]['post']

    my_env = os.environ.copy()
    my_env['subscription_id'] = subscription_id
    my_env['resource_group'] = resource_group
    my_env['workspace'] = workspace
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    submitted_job_list = []
    succeeded_jobs = []
    failed_jobs = []
    if group_pre:
        check_call(f"python {group_pre}", env=my_env, shell=True)

    with open(tests_dir / TEST_YML) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for job, job_data in data[test_group]['jobs'].items():
            if 'pre' in job_data:
                print(f"Running pre script for {job}")
                proc = check_call(f"python3 {tests_dir / job_data['pre']}", env=my_env, shell=True)
            print(f'Loading test job {job}')
            test_job = azure.ai.ml.load_job(tests_dir / job_data['job'])
            print(test_job)
            print(f'Running test job {job}')
            test_job = ml_client.jobs.create_or_update(test_job)
            print(f'Submitted test job {job}')
            print(f"Job id: {test_job.id}")
            submitted_job_list.append(test_job)

    # TO-DO: job post and group post scripts will be run after all jobs are completed

    while submitted_job_list:
        for job in submitted_job_list:
            returned_job = ml_client.jobs.get(job.name)
            print(f'The status of test job {job.name} is {returned_job.status}')
            if returned_job.status == "Completed":
                succeeded_jobs.append(returned_job.display_name)
                submitted_job_list.remove(job)
            elif returned_job.status == "Failed":
                failed_jobs.append(returned_job.display_name)
                submitted_job_list.remove(job)
    print(f"{len(succeeded_jobs) + len(failed_jobs)} jobs have been run. {len(succeeded_jobs)} jobs succeeded.")

    if failed_jobs:
        failed_job_str = ", ".join(failed_jobs)
        print(f"{len(failed_jobs)} jobs failed. {failed_job_str}.")
        sys.exit(1)
