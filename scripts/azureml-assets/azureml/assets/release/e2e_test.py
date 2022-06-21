from asyncio import subprocess
import argparse
from pathlib import Path
import os
import sys
import datetime
import ruamel.yaml
import subprocess
from subprocess import Popen,PIPE


# Handle command-line args
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input-dir", required=True, type=Path, help="dir path of tests folder")
args = parser.parse_args()
tests_dir = args.input_dir
final_report = {}

print("Start executing E2E tests script")
for area in os.listdir(tests_dir.__str__()):
    print(f"now processing area: {area}")
    final_report[area] = []
    yaml = ruamel.yaml.YAML()
    with open(tests_dir.__str__()+'/'+area+"/tests.yml") as fp:
        data = yaml.load(fp)
        for test_group in data:
            print(f"now processing test group: {test_group}")
            p = subprocess.Popen("python -u group_test.py -i "+tests_dir.__str__()+"/"+area+" -g "+test_group, stdout=PIPE, shell=True)
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
        print(area+ ': ' +group_test_report)
        if "failed" in group_test_report:
            red_flag = True

# fail the build if any test failed
if(red_flag):
    raise Exception("one or more test groups failed")