# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
# from asyncio import subprocess
import argparse
from pathlib import Path
import os
import yaml
from subprocess import Popen, PIPE
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential


# Handle command-line args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
parser.add_argument("-s", "--subscription", required=True, type=str, help="Subscription ID")
parser.add_argument("-r", "--resource-group", required=True, type=str, help="Resource group name")
parser.add_argument("-w", "--workspace-name", required=True, type=str, help="Workspace name")
args = parser.parse_args()
tests_dir = args.input_dir
subscription_id = args.subscription
resource_group = args.resource_group
workspace = args.workspace_name
final_report = {}
# subprocess
ml_client = MLClient(
    DefaultAzureCredential(), subscription_id, resource_group, workspace
)
cpu_cluster_low_pri = AmlCompute(
    name="cpu-cluster",
    size="Standard_DS3_v2",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
    tier="low_priority",
)
ml_client.begin_create_or_update(cpu_cluster_low_pri)
gpu_cluster_low_pri = AmlCompute(
    name="gpu-cluster",
    size="Standard_NC24",
    min_instances=0,
    max_instances=2,
    idle_time_before_scale_down=120,
    tier="low_priority",
)
ml_client.begin_create_or_update(gpu_cluster_low_pri)
print("Start executing E2E tests script")
for area in os.listdir(tests_dir.__str__()):
    print(f"now processing area: {area}")
    final_report[area] = []
    with open(tests_dir.__str__()+'/'+area+"/tests.yml") as fp:
        data = yaml.load(fp, Loader=yaml.FullLoader)
        for test_group in data:
            print(f"now processing test group: {test_group}")
            p = Popen("python3 -u group_test.py -i " + tests_dir.__str__() + "/" + area + " -g " + test_group + " -s " + subscription_id + " -r " + resource_group + " -w " + workspace, stdout=PIPE, shell=True)
            stdout = p.communicate()
            print(stdout[0].decode('utf-8'))
            final_report[area].append(stdout[0].decode('utf-8'))
            print(f"finished processing test group: {test_group}")
        print(f"finished processing area: {area}")

print("Finished all tests")
red_flag = False
for area in final_report:
    print(f"now printing report area: {area}")
    for group_test_report in final_report[area]:
        print(area + ': ' + group_test_report)
        if "failed" in group_test_report:
            red_flag = True

# fail the build if any test failed
if(red_flag):
    raise Exception("one or more test groups failed")
