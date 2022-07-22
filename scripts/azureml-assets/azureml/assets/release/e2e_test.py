"""python script used to run the end to end test jobs"""
# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
import argparse
from pathlib import Path
import sys
import yaml
from subprocess import run
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential

if __name__ == '__main__':
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
    for area in tests_dir.iterdir():
        print(f"now processing area: {area.name}")
        final_report[area.name] = []
        with open(area / "tests.yml") as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data:
                print(f"now processing test group: {test_group}")
                p = run(f"python3 -u group_test.py -i {area} -g {test_group} -s {subscription_id} -r {resource_group} "
                        f"-w {workspace}", shell=True)
                return_code = p.returncode
                print(return_code)
                final_report[area.name].append(f"test group {test_group} returned {return_code}")
                print(f"finished processing test group: {test_group}")
            print(f"finished processing area: {area.name}")

    print("Finished all tests")
    failures = False
    for area in final_report:
        print(f"now printing report area: {area}")
        for group_test_report in final_report[area]:
            print(area + ': ' + group_test_report)
            if group_test_report.endswith('1'):
                failures = True

    # fail the build if any test failed
    if failures:
        print("One or more tests failed")
        sys.exit(1)
