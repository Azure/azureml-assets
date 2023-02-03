# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Python script used to run the end to end test jobs."""
import argparse
from pathlib import Path
import sys
import yaml
from subprocess import run
from azure.ai.ml import MLClient
from azure.ai.ml.entities import AmlCompute
from azure.identity import DefaultAzureCredential


def e2e_test(
        tests_dir: Path,
        subscription_id: str,
        resource_group: str,
        workspace: str,
        coverage_report: Path = None,
        version_suffix: str = None):
    """Run end to end tests."""
    final_report = {}
    ml_client = MLClient(
        DefaultAzureCredential(), subscription_id, resource_group, workspace
    )
    cpu_cluster = AmlCompute(
        name="cpu-cluster",
        size="Standard_DS3_v2",
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120
    )
    gpu_cluster = AmlCompute(
        name="gpu-cluster",
        size="Standard_NC24",
        min_instances=0,
        max_instances=2,
        idle_time_before_scale_down=120
    )
    try:
        ml_client.begin_create_or_update(cpu_cluster)
        ml_client.begin_create_or_update(gpu_cluster)
    except Exception as e:
        print(f"Failed to create cluster: {e}")

    print("Start executing E2E tests script")
    for area in tests_dir.iterdir():
        print(f"now processing area: {area.name}")
        final_report[area.name] = []
        tests_yaml_file: Path = area / "tests.yml"
        if not tests_yaml_file.exists():
            print(f"Could not locate tests.yaml in {area}")
            continue
        with open(tests_yaml_file) as fp:
            data = yaml.load(fp, Loader=yaml.FullLoader)
            for test_group in data:
                print(f"now processing test group: {test_group}")
                cmd = ["python3", "-u", "group_test.py", "-i", area, "-g", test_group, "-s", subscription_id,
                       "-r", resource_group, "-w", workspace]
                if coverage_report:
                    cmd.extend(["-c", coverage_report])
                if version_suffix:
                    cmd.extend(["-v", version_suffix])
                p = run(cmd)
                return_code = p.returncode
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


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
    parser.add_argument("-s", "--subscription", required=True, type=str, help="Subscription ID")
    parser.add_argument("-r", "--resource-group", required=True, type=str, help="Resource group name")
    parser.add_argument("-w", "--workspace-name", required=True, type=str, help="Workspace name")
    parser.add_argument("-c", "--coverage-report", required=False, type=Path, help="Path of coverage report yaml")
    parser.add_argument("-v", "--version-suffix", required=False, type=str,
                        help="version suffix which will be used to identify the asset id in tests")
    args = parser.parse_args()
    e2e_test(args.input_dir, args.subscription, args.resource_group, args.workspace_name,
             args.coverage_report, args.version_suffix)
