import argparse
from asyncio import subprocess
import subprocess
from pathlib import Path
import azure.ai.ml 
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
import ruamel.yaml
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
parser.add_argument("-g", "--test-group", required=True, type=str, help="test group name")
args = parser.parse_args()
tests_dir = args.input_dir
test_group = args.test_group
# default workspace info
workspace_name: "registry-builtin-ci-dev"
resource_group_name: "registry-builtin-dev"
subscription_id: "4f26493f-21d2-4726-92ea-1ddd550b1d27"
group_pre = ''
group_post = ''

yaml = ruamel.yaml.YAML()
with open(tests_dir.__str__()+"/tests.yml") as fp:
    data = yaml.load(fp)
    if 'subscription_id' in data[test_group]:
        subscription_id = data[test_group]['subscription_id']
    if 'resource_group_name' in data[test_group]:
        resource_group_name = data[test_group]['resource_group_name']
    if 'workspace_name' in data[test_group]:
        workspace_name = data[test_group]['workspace_name']
    if 'pre' in data[test_group]:
        group_pre = tests_dir.__str__()+'/'+data[test_group]['pre']
    if 'post' in data[test_group]:
        group_post = tests_dir.__str__()+'/'+data[test_group]['post']
    
ml_client = MLClient(DefaultAzureCredential(), subscription_id, resource_group_name, workspace_name)
submitted_job_list = []
succeeded_jobs = []
failed_jobs = []
subprocess.check_call(f"python {group_pre}", shell=True)

with open(tests_dir.__str__()+"/tests.yml") as fp:
    data = yaml.load(fp)
    for job in data[test_group]['jobs']:
        if 'pre' in data[test_group]['jobs'][job]:
            print(f"Running pre script for {job}")
            proc = subprocess.Popen(f"python {tests_dir.__str__()+'/'+data[test_group]['jobs'][job]['pre']}", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
            proc.wait()
            (stdout, stderr) = proc.communicate()
            if proc.returncode != 0:
                print(stderr)
                sys.exit(f"pre script of {job} is failed")
        print(f'Loading test job {job}')
        test_job = azure.ai.ml.load_job(tests_dir.__str__()+"/"+data[test_group]['jobs'][job]['job'])
        print(test_job)
        print(f'Running test job {job}')
        test_job = ml_client.jobs.create_or_update(test_job)
        print(f'Submitted test job {job}')
        print(f"Job id: {test_job.id}")
        submitted_job_list.append(test_job)

# TO-DO: job post and group post scripts will be run after all jobs are completed

while(len(submitted_job_list)):
    for job in submitted_job_list:
        returned_job = ml_client.jobs.get(job.name)
        print(f'The status of test job {job.name} is {returned_job.status}')
        if(returned_job.status == "Completed"):
            succeeded_jobs.append(returned_job.display_name)
            submitted_job_list.remove(job)
        elif(returned_job.status == "Failed"):
            failed_jobs.append(returned_job.display_name)
            submitted_job_list.remove(job)
        
print(f"Totally {len(succeeded_jobs)+len(failed_jobs)} jobs have been run. {len(succeeded_jobs)} jobs succeeded.")

#
if(len(failed_jobs)>0):
    failed_job_str = ", ".join(failed_jobs)
    print(f"{len(failed_jobs)} jobs failed. {failed_job_str}.")

