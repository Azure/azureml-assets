# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import Dict, List, Tuple

FLAKE_RULES_FILE = "flake_rules.json"
DEFAULT_MAX_LINE_LENGTH = 10000
IGNORE = "ignore"
IGNORE_FILE = "ignore-file"
EXCLUDE = "exclude"
MAX_LINE_LENGTH = "max-line-length"


def run_flake8(testpath: Path, flake_rules: Dict[str, List[str]]) -> Tuple[int, str]:
    ignore = flake_rules.get(IGNORE, [])
    file_ignore = flake_rules.get(IGNORE_FILE, [])
    exclude = flake_rules.get(EXCLUDE, [])
    max_line_length = flake_rules.get(MAX_LINE_LENGTH, DEFAULT_MAX_LINE_LENGTH)
    cmd = [
        "flake8",
        f"--max-line-length={max_line_length}",
        testpath
    ]
    if exclude:
        cmd.insert(1, "--exclude={}".format(",".join(exclude)))
    if file_ignore:
        cmd.insert(1, "--per-file-ignores={}".format(",".join(file_ignore)))
    if ignore:
        cmd.insert(1, "--ignore={}".format(",".join(ignore)))

    print(f"Running {cmd}")
    p = run(cmd,
            stdout=PIPE,
            stderr=STDOUT)

    return p.stdout.decode()


def load_rules(testpath: Path) -> Dict[str, List[str]]:
    flake_rules_file = testpath / FLAKE_RULES_FILE
    flake_rules = {}
    if flake_rules_file.exists():
        with open(flake_rules_file) as f:
            flake_rules = json.load(f).get('pep8', {})
    
    # Handle relative paths
    if IGNORE_FILE in flake_rules:
        file_ignore = []
        for pair in flake_rules[IGNORE_FILE]:
            file, rules = pair.split(":")
            file_resolved = str(testpath / file)
            file_ignore.append(f"{file_resolved}:{rules}")
        flake_rules[IGNORE_FILE] = file_ignore
    if EXCLUDE in flake_rules:
        flake_rules[EXCLUDE] = [str(testpath / p) for p in flake_rules[EXCLUDE]]

    return flake_rules


def combine_rules(rule_a: Dict[str, List[str]], rule_b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    rule = {}
    rule[IGNORE] = rule_a.get(IGNORE, []) + rule_b.get(IGNORE, [])
    rule[IGNORE_FILE] = rule_a.get(IGNORE_FILE, []) + rule_b.get(IGNORE_FILE, [])
    rule[EXCLUDE] = rule_a.get(EXCLUDE, []) + rule_b.get(EXCLUDE, [])
    rule[MAX_LINE_LENGTH] = min(rule_a.get(MAX_LINE_LENGTH, DEFAULT_MAX_LINE_LENGTH),
                                rule_b.get(MAX_LINE_LENGTH, DEFAULT_MAX_LINE_LENGTH))

    return rule


def inherit_flake_rules(rootpath: Path, testpath: Path) -> Dict[str, List[str]]:
    flake_rules = {}
    upperpath = testpath
    while upperpath != rootpath:
        upperpath = upperpath.parent
        flake_rules = combine_rules(flake_rules, load_rules(upperpath))
    return flake_rules


def test(rootpath: Path, testpath: Path):
    test_path_flake_rules = inherit_flake_rules(rootpath, testpath)

    flake_rules_files = testpath.rglob(FLAKE_RULES_FILE)
    custom = [p.parent for p in flake_rules_files]

    if testpath not in custom:
        custom.append(testpath)

    output = []
    for path in custom:
        flake_rules = {}
        flake_rules[EXCLUDE] = [p.parent for p in flake_rules_files if p.parent != path]
        inherited_rules = inherit_flake_rules(testpath, path)
        flake_rules = combine_rules(combine_rules(flake_rules, test_path_flake_rules),
                                    combine_rules(inherited_rules, load_rules(path)))
        output.extend([line for line in run_flake8(path, flake_rules).split("\n") if len(line) > 0])

    if len(output) > 0:
        print("flake8 errors:")
        for line in output:
            print(line)
        raise Exception("Code is unhealthy.")


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, type=Path, help="Directory to validate")
    parser.add_argument("-r", "--root-directory", type=Path, help="Root directory containing flake8 rules, must be a parent of --input-directory")
    args = parser.parse_args()

    # Handle directories
    input_directory = args.input_directory
    root_directory = args.root_directory
    if root_directory is None:
        root_directory = args.input_directory
    elif not input_directory.is_relative_to(root_directory):
        parser.error(f"{root_directory} is not a parent directory of {input_directory}")

    test(root_directory, input_directory)
