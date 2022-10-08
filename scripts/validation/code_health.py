# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Check code health using flake8."""

import argparse
import json
import sys
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import Dict, List, Set, Tuple

RULES_FILENAME = "validation_rules.json"


class _Rules:
    ROOT_KEY = "pep8"
    IGNORE = "ignore"
    IGNORE_FILE = "ignore-file"
    EXCLUDE = "exclude"
    MAX_LINE_LENGTH = "max-line-length"
    DEFAULT_MAX_LINE_LENGTH = 10000

    def __init__(self, file_name: Path = None):
        if file_name is not None and file_name.exists():
            parent_path = file_name.parent
            # Load rules from file
            with open(file_name) as f:
                rules = json.load(f).get(self.ROOT_KEY, {})
                self.ignore = set(rules.get(self.IGNORE, []))
                self.ignore_file = self._parse_ignore_file(parent_path, rules.get(self.IGNORE_FILE, []))
                self.exclude = {parent_path / p for p in rules.get(self.EXCLUDE, [])}
                self.max_line_length = rules.get(self.MAX_LINE_LENGTH)
        else:
            # Initialize empty
            self.ignore = set()
            self.ignore_file = {}
            self.exclude = set()
            self.max_line_length = None

    @staticmethod
    def _parse_ignore_file(parent_path: Path, ignore_file: List[str]) -> Dict[str, Set[str]]:
        results = {}
        for pair in ignore_file:
            file, file_rules = pair.split(":")
            file = parent_path / file
            file_rules = {r.strip() for r in file_rules.split(",") if not r.isspace()}
            results[file] = file_rules
        return results

    def get_effective_max_line_length(self) -> int:
        return self.max_line_length or self.DEFAULT_MAX_LINE_LENGTH

    def __or__(self, other: "_Rules") -> "_Rules":
        rules = _Rules()
        rules.ignore = self.ignore | other.ignore
        for key in self.ignore_file.keys() | other.ignore_file.keys():
            rules.ignore_file[key] = self.ignore_file.get(key, set()) | other.ignore_file.get(key, set())
        rules.exclude = self.exclude | other.exclude
        rules.max_line_length = other.max_line_length or self.max_line_length
        return rules


def _run_flake8(testpath: Path, rules: _Rules) -> Tuple[int, str]:
    cmd = [
        "flake8",
        f"--max-line-length={rules.get_effective_max_line_length()}",
        str(testpath)
    ]
    if rules.exclude:
        cmd.insert(1, "--exclude={}".format(",".join([str(e) for e in rules.exclude])))
    if rules.ignore_file:
        file_ignore_list = []
        for file, ignores in rules.ignore_file.items():
            file_ignore_list.extend([f"{file}:{i}" for i in ignores])
        cmd.insert(1, "--per-file-ignores={}".format(",".join(file_ignore_list)))
    if rules.ignore:
        cmd.insert(1, "--ignore={}".format(",".join(rules.ignore)))

    print(f"Running {cmd}")
    p = run(cmd,
            stdout=PIPE,
            stderr=STDOUT)

    return p.stdout.decode()


def _inherit_rules(rootpath: Path, testpath: Path) -> _Rules:
    # Process paths from rootpath to testpath, to ensure max_line_length is calculated properly
    paths = [p for p in testpath.parents if p == rootpath or p.is_relative_to(rootpath)]
    paths.reverse()
    rules = _Rules()
    for path in paths:
        rules |= _Rules(path / RULES_FILENAME)
    return rules


def _test(rootpath: Path, testpath: Path) -> bool:
    testpath_rules = _inherit_rules(rootpath, testpath)

    rules_files = list(testpath.rglob(RULES_FILENAME))
    dirs = [p.parent for p in rules_files]

    if testpath not in dirs:
        dirs.insert(0, testpath)

    errors = []
    for path in dirs:
        rules = _Rules()
        rules.exclude = {d for d in dirs if d != path and not path.is_relative_to(d)}
        rules |= testpath_rules | _inherit_rules(testpath, path) | _Rules(path / RULES_FILENAME)
        errors.extend([line for line in _run_flake8(path, rules).splitlines() if len(line) > 0])

    if len(errors) > 0:
        print("flake8 errors:")
        for line in errors:
            print(line)
        return False
    return True


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directory", required=True, type=Path,
                        help="Directory to validate")
    parser.add_argument("-r", "--root-directory", type=Path,
                        help="Root directory containing flake8 rules, must be a parent of --input-directory")
    args = parser.parse_args()

    # Handle directories
    input_directory = args.input_directory
    root_directory = args.root_directory
    if root_directory is None:
        root_directory = args.input_directory
    elif not input_directory.is_relative_to(root_directory):
        parser.error(f"{root_directory} is not a parent directory of {input_directory}")

    success = _test(root_directory, input_directory)

    if not success:
        sys.exit(1)
