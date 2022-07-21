# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""Check docstrings using pydocstyle."""

import argparse
import json
import re
import sys
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import List, Set

RULES_FILENAME = "validation_rules.json"
FILE_NAME_PATTERN = re.compile(r"^(.+):\d+\s+")


class _Rules:
    ROOT_KEY = "doc"
    IGNORE = "ignore"
    EXCLUDE = "exclude"
    FORCE = "force"

    def __init__(self, file_name: Path = None):
        if file_name is not None and file_name.exists():
            # Load rules from file
            with open(file_name) as f:
                rules = json.load(f).get(self.ROOT_KEY, {})
                self.ignore = set(rules.get(self.IGNORE, []))
                self.exclude = {file_name.parent / p for p in rules.get(self.EXCLUDE, [])}
                self.force = set(rules.get(self.FORCE, []))
        else:
            # Initialize empty
            self.ignore = set()
            self.exclude = set()
            self.force = set()

    def __or__(self, other: "_Rules") -> "_Rules":
        rules = _Rules()
        rules.ignore = self.ignore | other.ignore
        rules.exclude = self.exclude | other.exclude
        rules.force = self.force | other.force
        return rules


def _run_docstyle(testpath: Path, rules: _Rules):
    cmd = [
        "pydocstyle",
        r"--match=.*\.py",
        str(testpath)
    ]

    ignore = rules.ignore - rules.force
    if ignore:
        cmd.insert(1, "--ignore={}".format(",".join(ignore)))

    print(f"Running {cmd}")
    p = run(cmd,
            stdout=PIPE,
            stderr=STDOUT)

    return p.stdout.decode()


def _filter_docstyle_output(output: str, rules: _Rules, changed_files: List[Path]) -> List[str]:
    lines = [line for line in output.splitlines() if len(line) > 0]
    if lines:
        # Iterate over every other line, since pydocstyle splits output across two lines
        filtered_lines = []
        for i in range(0, len(lines) - 1, 2):
            # Parse filename
            line = lines[i]
            match = FILE_NAME_PATTERN.match(line)
            if not match:
                raise Exception(f"Unable to extract filename from {line}")
            file = Path(match.group(1))

            # Determine whether file should be included in output
            include_file = True
            for exclude in rules.exclude:
                # Check if file is explicitly excluded or in an excluded dir
                if file == exclude or (exclude.is_dir() and file.is_relative_to(exclude)):
                    include_file = False
                    break

            # Include only changed files, if specified
            if include_file and changed_files is not None and file not in changed_files:
                include_file = False

            # Store lines if file is included
            if include_file:
                filtered_lines.append(line)
                filtered_lines.append(lines[i + 1])
        lines = filtered_lines

    return lines


def _inherit_rules(rootpath: Path, testpath: Path) -> _Rules:
    rules = _Rules()
    upperpath = testpath
    while upperpath != rootpath:
        upperpath = upperpath.parent
        rules |= _Rules(upperpath / RULES_FILENAME)
    return rules


def _test(rootpath: Path, testpath: Path, force: Set[str], changed_files: List[Path]) -> bool:
    testpath_rules = _inherit_rules(rootpath, testpath)

    rules_files = list(testpath.rglob(RULES_FILENAME))
    dirs = [p.parent for p in rules_files]

    if testpath not in dirs:
        dirs.insert(0, testpath)

    errors = []
    for path in dirs:
        rules = _Rules()
        rules.exclude = {d for d in dirs if d != path and not path.is_relative_to(d)}
        rules.force = force
        rules |= testpath_rules | _inherit_rules(testpath, path) | _Rules(path / RULES_FILENAME)

        output = _run_docstyle(path, rules)
        filtered_output = _filter_docstyle_output(output, rules, changed_files)
        errors.extend(filtered_output)

    if len(errors) > 0:
        print("pydocstyle errors:")
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
                        help="Root directory containing docstyle rules, must be a parent of --input-directory")
    parser.add_argument("-f", "--force", default="",
                        help="Comma-separated list of rules that can't be ignored")
    parser.add_argument("-c", "--changed-files",
                        help="Comma-separated list of changed files, used to filter assets")
    args = parser.parse_args()

    # Handle directories
    input_directory = args.input_directory
    root_directory = args.root_directory
    if root_directory is None:
        root_directory = args.input_directory
    elif not input_directory.is_relative_to(root_directory):
        parser.error(f"{root_directory} is not a parent directory of {input_directory}")

    # Convert comma-separated values to lists
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else None

    # Parse forced rules
    force = {r.strip() for r in args.force.split(",") if not r.isspace()}

    success = _test(rootpath=root_directory,
                   testpath=input_directory,
                   force=force,
                   changed_files=changed_files)

    if not success:
        sys.exit(1)
