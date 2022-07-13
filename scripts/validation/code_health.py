# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import json
import sys
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import Dict, List, Tuple

VALIDATION_RULES_FILE = "validation_rules.json"
DEFAULT_MAX_LINE_LENGTH = 10000
IGNORE = "ignore"
IGNORE_FILE = "ignore-file"
EXCLUDE = "exclude"
MAX_LINE_LENGTH = "max-line-length"


def run_flake8(testpath: Path, rules: Dict[str, List[str]]) -> Tuple[int, str]:
    ignore = rules.get(IGNORE, [])
    file_ignore = rules.get(IGNORE_FILE, [])
    exclude = rules.get(EXCLUDE, [])
    max_line_length = rules.get(MAX_LINE_LENGTH, DEFAULT_MAX_LINE_LENGTH)
    cmd = [
        "flake8",
        f"--max-line-length={max_line_length}",
        str(testpath)
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
    rules_file = testpath / VALIDATION_RULES_FILE
    rules = {}
    if rules_file.exists():
        with open(rules_file) as f:
            rules = json.load(f).get('pep8', {})

    # Handle relative paths
    if IGNORE_FILE in rules:
        ignore_file = []
        for pair in rules[IGNORE_FILE]:
            file, file_rules = pair.split(":")
            file_resolved = str(testpath / file)
            ignore_file.append(f"{file_resolved}:{file_rules}")
        rules[IGNORE_FILE] = ignore_file
    if EXCLUDE in rules:
        rules[EXCLUDE] = [str(testpath / p) for p in rules[EXCLUDE]]

    return rules


def combine_rules(rule_a: Dict[str, List[str]], rule_b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    rule = {}
    rule[IGNORE] = rule_a.get(IGNORE, []) + rule_b.get(IGNORE, [])
    rule[IGNORE_FILE] = rule_a.get(IGNORE_FILE, []) + rule_b.get(IGNORE_FILE, [])
    rule[EXCLUDE] = rule_a.get(EXCLUDE, []) + rule_b.get(EXCLUDE, [])
    rule[MAX_LINE_LENGTH] = min(rule_a.get(MAX_LINE_LENGTH, DEFAULT_MAX_LINE_LENGTH),
                                rule_b.get(MAX_LINE_LENGTH, DEFAULT_MAX_LINE_LENGTH))

    return rule


def inherit_rules(rootpath: Path, testpath: Path) -> Dict[str, List[str]]:
    rules = {}
    upperpath = testpath
    while upperpath != rootpath:
        upperpath = upperpath.parent
        rules = combine_rules(rules, load_rules(upperpath))
    return rules


def test(rootpath: Path, testpath: Path) -> bool:
    testpath_rules = inherit_rules(rootpath, testpath)

    rules_files = list(testpath.rglob(VALIDATION_RULES_FILE))
    dirs = [p.parent for p in rules_files]

    if testpath not in dirs:
        dirs.insert(0, testpath)

    errors = []
    for path in dirs:
        rules = {}
        rules[EXCLUDE] = [str(d) for d in dirs if d != path and not path.is_relative_to(d)]
        inherited_rules = inherit_rules(testpath, path)
        rules = combine_rules(combine_rules(rules, testpath_rules),
                              combine_rules(inherited_rules, load_rules(path)))
        errors.extend([line for line in run_flake8(path, rules).splitlines() if len(line) > 0])

    if len(errors) > 0:
        print("flake8 errors:")
        for line in errors:
            print(line)
        return False
    return True


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

    success = test(root_directory, input_directory)

    if not success:
        sys.exit(1)
