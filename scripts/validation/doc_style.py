import argparse
import json
import re
import sys
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import List, Set

RULES_FILENAME = "validation_rules.json"
FILE_NAME_PATTERN = re.compile(r"^(.+):\d+\s+")


class Rules:
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

    def __or__(self, other: "Rules") -> "Rules":
        rules = Rules()
        rules.ignore = self.ignore | other.ignore
        rules.exclude = self.exclude | other.exclude
        rules.force = self.force | other.force
        return rules


def run_docstyle(testpath: Path, rules: Rules):
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


def filter_docstyle_output(output: str, rules: Rules) -> List[str]:
    lines = [line for line in output.splitlines() if len(line) > 0]
    if lines:
        filtered_lines = []
        for i in range(0, len(lines) - 1, 2):
            line = lines[i]
            match = FILE_NAME_PATTERN.match(line)
            if not match:
                raise Exception(f"Unable to extract filename from {line}")
            file = Path(match.group(1))
            file_is_excluded = False
            for exclude in rules.exclude:
                if file == exclude or (exclude.is_dir() and file.is_relative_to(exclude)):
                    file_is_excluded = True
                    break
            if not file_is_excluded:
                filtered_lines.append(line)
                filtered_lines.append(lines[i + 1])
        lines = filtered_lines

    return lines


def inherit_rules(rootpath: Path, testpath: Path) -> Rules:
    rules = Rules()
    upperpath = testpath
    while upperpath != rootpath:
        upperpath = upperpath.parent
        rules |= Rules(upperpath / RULES_FILENAME)
    return rules


def test(rootpath: Path, testpath: Path, force: Set[str] = {}) -> bool:
    testpath_rules = inherit_rules(rootpath, testpath)

    rules_files = list(testpath.rglob(RULES_FILENAME))
    dirs = [p.parent for p in rules_files]

    if testpath not in dirs:
        dirs.insert(0, testpath)

    errors = []
    for path in dirs:
        rules = Rules()
        rules.exclude = {d for d in dirs if d != path and not path.is_relative_to(d)}
        rules.force = force
        rules |= testpath_rules | inherit_rules(testpath, path) | Rules(path / RULES_FILENAME)

        output = run_docstyle(path, rules)
        filtered_output = filter_docstyle_output(output, rules)
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
    parser.add_argument("-i", "--input-directory", required=True, type=Path, help="Directory to validate")
    parser.add_argument("-r", "--root-directory", type=Path, help="Root directory containing docstyle rules, must be a parent of --input-directory")
    parser.add_argument("-f", "--force", default="", help="Comma-separated list of rules that can't be ignored")
    args = parser.parse_args()

    # Handle directories
    input_directory = args.input_directory
    root_directory = args.root_directory
    if root_directory is None:
        root_directory = args.input_directory
    elif not input_directory.is_relative_to(root_directory):
        parser.error(f"{root_directory} is not a parent directory of {input_directory}")

    # Parse forced rules
    force = {r.strip() for r in args.force.split(",") if not r.isspace()}

    success = test(root_directory, input_directory, force)

    if not success:
        sys.exit(1)
