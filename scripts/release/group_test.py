# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python script to run group tests."""
import argparse
from subprocess import check_call, run
from pathlib import Path
import azure.ai.ml
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import concurrent.futures
import yaml
import os
import sys
import logging

logger = logging.getLogger(__name__)
TEST_YML = "tests.yml"


def run_pytest_job(job: Path, my_env: dict):
    """Run single pytest job."""
    p = run(f"pytest -q {job}", env=my_env, shell=True)
    return p.returncode


def run_pytest_jobs(pytest_jobs: dict, my_env: dict):
    """Run multiple pytest jobs concurrently."""
    logger.info("Start running pytest jobs")
    jobs = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        futures_to_job = {executor.submit(run_pytest_job, job, my_env): job for job in pytest_jobs.keys()}
        for future in concurrent.futures.as_completed(futures_to_job):
            job = futures_to_job[future]
            jobs[future] = pytest_jobs[job]
    return jobs


def group_test(
        tests_dir: Path,
        test_group: str,
        subscription_id: str,
        resource_group: str,
        workspace: str,
        coverage_report: Path = None,
        version_suffix: str = None):
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
    if my_env['token']:
        logger.info("token is set")
    if version_suffix:
        my_env['version_suffix'] = version_suffix
    ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group, workspace)
    submitted_job_list = []
    succeeded_jobs = []
    failed_jobs = []
    test_coverage = {}
    covered_assets = []

    if group_pre:
        check_call(f"python {group_pre}", env=my_env, shell=True)

    pytest_jobs = {}  # pytest job path -> assets coverage dict
    with open(tests_dir / TEST_YML) as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for job, job_data in data[test_group]['jobs'].items():
            if 'pytest_job' in job_data:
                pytest_jobs[tests_dir / job_data['pytest_job']] = job_data['assets']
            else:
                if 'pre' in job_data:
                    logger.info(f"Running pre script for {job}")
                    proc = check_call(f"python3 {tests_dir / job_data['pre']}", env=my_env, shell=True)
                print(f'Loading test job {job}')
                try:
                    test_job = azure.ai.ml.load_job(tests_dir / job_data['job'])
                    logger.info(f'Running test job {job}')
                    test_job = ml_client.jobs.create_or_update(test_job)
                except Exception as ex:
                    logger.warning(
                        f"catch error submitting {job_data['job']} with exception {ex}")
                    failed_jobs.append(job)
                    continue
                test_coverage[test_job] = job_data.get("assets", [])
                logger.info(f'Submitted test job {job}')
                logger.info(f"Job id: {test_job.id}")
                submitted_job_list.append(test_job)

    # Run pytest jobs
    pytest_job_results = run_pytest_jobs(pytest_jobs, my_env)

    for job, assets in pytest_job_results.items():
        if job.result() == 0:
            succeeded_jobs.append(job)
            covered_assets.extend(assets)
        else:
            failed_jobs.append(job)

    # TO-DO: job post and group post scripts will be run after all jobs are completed

    # process pipeline jobs results
    while submitted_job_list:
        for job in submitted_job_list:
            returned_job = ml_client.jobs.get(job.name)
            if returned_job.status == "Completed":
                succeeded_jobs.append(returned_job.display_name)
                covered_assets.extend(test_coverage.get(job, []))
                submitted_job_list.remove(job)
            elif returned_job.status in ["Failed", "Cancelled"]:
                failed_jobs.append(returned_job.display_name)
                submitted_job_list.remove(job)

    print(f"{len(succeeded_jobs) + len(failed_jobs)} jobs have been run. {len(succeeded_jobs)} jobs succeeded.")

    logger.info(f"covered_assets {covered_assets}")
    if coverage_report:
        with open(coverage_report, 'r') as yf:
            cover_yaml = yaml.safe_load(yf) or []
            cover_yaml.extend(covered_assets)
        with open(coverage_report, 'w') as yf:
            yaml.safe_dump(cover_yaml, yf)

    if failed_jobs:
        logger.warning(f"{len(failed_jobs)} jobs failed. {failed_jobs}.")
        sys.exit(1)


if __name__ == '__main__':
    logger.info("Start running group tests")
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
    parser.add_argument("-g", "--test-group", required=True, type=str, help="test group name")
    parser.add_argument("-s", "--subscription", required=True, type=str, help="Subscription ID")
    parser.add_argument("-r", "--resource-group", required=True, type=str, help="Resource group name")
    parser.add_argument("-w", "--workspace-name", required=True, type=str, help="Workspace name")
    parser.add_argument("-c", "--coverage-report", required=False, type=Path, help="Path of coverage report yaml")
    parser.add_argument("-v", "--version-suffix", required=False, type=str,
                        help="version suffix which will be used to identify the asset id in tests")
    args = parser.parse_args()
    group_test(args.input_dir, args.test_group, args.subscription, args.resource_group, args.workspace_name,
               args.coverage_report, args.version_suffix)
