# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python script to run group tests."""
import argparse
from subprocess import check_call
from subprocess import run
from pathlib import Path
import azure.ai.ml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from collections import defaultdict
import concurrent.futures
import yaml
import os
import sys
from azureml.assets.util import logger
TEST_YML = "tests.yml"


def run_pytest_job(job: Path, my_env: dict):
    """Run single pytest job."""
    p = run(f"pytest -q {job}", env=my_env, shell=True)
    return_code = p.returncode
    return return_code


def run_pytest_jobs_old(pytest_jobs: dict, my_env: dict):
    """Run multiple pytest jobs concurrently."""
    logger.print("Start running pytest jobs")
    submitted_jobs = defaultdict(list)
    executor = concurrent.futures.ThreadPoolExecutor()
    for job in pytest_jobs.keys():
        arr = [job, my_env]
        future = executor.submit(lambda p: run_pytest_job(*p), arr)
        submitted_jobs[future] = pytest_jobs[job]
    return submitted_jobs

def run_pytest_jobs(pytest_jobs: dict, my_env: dict):
    """Run multiple pytest jobs concurrently."""
    logger.print("Start running pytest jobs")
    submitted_jobs = defaultdict(list)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(run_pytest_job, job, my_env) for job in pytest_jobs.keys()]
        for future in concurrent.futures.as_completed(futures):
            submitted_jobs[future] = pytest_jobs[job]
    return submitted_jobs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
    parser.add_argument("-g", "--test-group", required=True, type=str, help="test group name")
    parser.add_argument("-s", "--subscription", required=True, type=str, help="Subscription ID")
    parser.add_argument("-r", "--resource-group", required=True, type=str, help="Resource group name")
    parser.add_argument("-w", "--workspace-name", required=True, type=str, help="Workspace name")
    parser.add_argument("-c", "--coverage-report", required=False, type=Path, help="Path of coverage report yaml")
    parser.add_argument("-v", "--version-suffix", required=False, type=str, help="version-suffix")
    parser.add_argument("-t", "--token", required=True, type=str, help="the Bearer token")
    args = parser.parse_args()
    tests_dir = args.input_dir
    test_group = args.test_group
    subscription_id = args.subscription
    resource_group = args.resource_group
    workspace = args.workspace_name
    coverage_report = args.coverage_report

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
    my_env['token'] = args.token.strip('"')
    if args.version_suffix:
        my_env['version_suffix'] = args.version_suffix
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    submitted_job_list = []
    succeeded_jobs = []
    failed_jobs = []
    test_coverage = defaultdict(list)
    covered_assets = []

    if group_pre:
        check_call(f"python {group_pre}", env=my_env, shell=True)

    pytest_jobs = defaultdict(list)  # pytest job path -> assets coverage dict
    with open(tests_dir / TEST_YML) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for job, job_data in data[test_group]['jobs'].items():
            if 'pytest_job' in job_data:
                pytest_jobs[tests_dir / job_data['pytest_job']] = job_data["assets"]
            else:
                if 'pre' in job_data:
                    logger.print(f"Running pre script for {job}")
                    proc = check_call(f"python3 {tests_dir / job_data['pre']}", env=my_env, shell=True)
                print(f'Loading test job {job}')
                try:
                    test_job = azure.ai.ml.load_job(tests_dir / job_data['job'])
                    logger.print(f'Running test job {job}')
                    test_job = ml_client.jobs.create_or_update(test_job)
                except Exception as ex:
                    logger.log_warning(
                        f"catch error submitting {job_data['job']} with exception {ex}")
                    failed_jobs.append(job)
                    continue
                test_coverage[test_job] = job_data.get("assets", [])
                logger.print(f'Submitted test job {job}')
                logger.print(f"Job id: {test_job.id}")
                submitted_job_list.append(test_job)

    # Run pytest jobs
    submitted_pytest_jobs = run_pytest_jobs(pytest_jobs, my_env)

    # Process pytest results
    # while submitted_pytest_jobs:
    #    for job in submitted_pytest_jobs.copy().keys():
    #        if job.done():
    #            if job.result() == 0:
    #                succeeded_jobs.append(job)
    #                covered_assets.extend(submitted_pytest_jobs[job])
    #            else:
    #                failed_jobs.append(job)
    #            del submitted_pytest_jobs[job]

    for job in submitted_pytest_jobs.keys():
        if job.result() == 0:
            succeeded_jobs.append(job)
            covered_assets.extend(submitted_pytest_jobs[job])
        else:
            failed_jobs.append(job)
            
    # TO-DO: job post and group post scripts will be run after all jobs are completed

    # process pipeline jobs results
    while submitted_job_list:
        for job in submitted_job_list:
            returned_job = ml_client.jobs.get(job.name)
            if returned_job.status == "Completed":
                succeeded_jobs.append(returned_job.display_name)
                submitted_job_list.remove(job)
                covered_assets.extend(test_coverage.get(job, []))
            elif returned_job.status == "Failed" or returned_job.status == "Cancelled":
                failed_jobs.append(returned_job.display_name)
                submitted_job_list.remove(job)

    print(f"{len(succeeded_jobs) + len(failed_jobs)} jobs have been run. {len(succeeded_jobs)} jobs succeeded.")

    logger.print(f"covered_assets {covered_assets}")
    if coverage_report:
        with open(coverage_report, 'r') as yf:
            cover_yaml = yaml.safe_load(yf)
            if not cover_yaml:
                cover_yaml = []
            cover_yaml.extend(covered_assets)
        with open(coverage_report, 'w') as yf:
            yaml.safe_dump(cover_yaml, yf)

    if failed_jobs:
        logger.log_warning(f"{len(failed_jobs)} jobs failed. {failed_jobs}.")
        sys.exit(1)
