import argparse
import json
import re
import sys
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from typing import Dict, List

VALIDATION_RULES_FILE = "validation_rules.json"
IGNORE = "ignore"
EXCLUDE = "exclude"
FILE_NAME_PATTERN = re.compile(r"^(.+):\d+\s+")


def run_docstyle(testpath: Path, rules: Dict[str, List[str]]):
    ignore = rules.get(IGNORE, [])

    cmd = [
        "pydocstyle",
        r"--match=.*\.py",
        testpath
    ]

    if ignore:
        cmd.insert(1, "--ignore={}".format(",".join(ignore)))

    print(f"Running {cmd}")
    p = run(cmd,
            stdout=PIPE,
            stderr=STDOUT)

    return p.stdout.decode()


def filter_docstyle_output(output: str, rules: Dict[str, List[str]]) -> List[str]:
    lines = [line for line in output.splitlines() if len(line) > 0]
    if lines:
        excludes = [Path(e) for e in rules[EXCLUDE]]
        filtered_lines = []
        for i in range(0, len(lines) - 1, 2):
            line = lines[i]
            match = FILE_NAME_PATTERN.match(line)
            if not match:
                raise Exception(f"Unable to extract filename from {line}")
            file = Path(match.group(1))
            file_is_excluded = False
            for exclude in excludes:
                if file == exclude or (exclude.is_dir() and file.is_relative_to(exclude)):
                    file_is_excluded = True
                    break
            if not file_is_excluded:
                filtered_lines.append(line)
                filtered_lines.append(lines[i + 1])
        lines = filtered_lines

    return lines


def load_rules(testpath: Path, force: List[str] = []) -> Dict[str, List[str]]:
    validation_rules_file = testpath / VALIDATION_RULES_FILE
    rules = {}
    if validation_rules_file.exists():
        with open(validation_rules_file) as f:
            rules = json.load(f).get('doc', {})

    # Filter out any ignored rules that are forced
    if IGNORE in rules and force:
        rules[IGNORE] = [r for r in rules[IGNORE] if r not in force]

    # Handle relative paths
    if EXCLUDE in rules:
        rules[EXCLUDE] = [str(testpath / p) for p in rules[EXCLUDE]]

    return rules


def combine_rules(rule_a: Dict[str, List[str]], rule_b: Dict[str, List[str]]) -> Dict[str, List[str]]:
    rule = {}
    rule[IGNORE] = rule_a.get(IGNORE, []) + rule_b.get(IGNORE, [])
    rule[EXCLUDE] = rule_a.get(EXCLUDE, []) + rule_b.get(EXCLUDE, [])

    return rule


def inherit_rules(rootpath: Path, testpath: Path, force: List[str] = []) -> Dict[str, List[str]]:
    rules = {}
    upperpath = testpath
    while upperpath != rootpath:
        upperpath = upperpath.parent
        rules = combine_rules(rules, load_rules(upperpath, force))
    return rules


def test(rootpath: Path, testpath: Path, force: List[str] = []) -> bool:
    testpath_rules = inherit_rules(rootpath, testpath, force)

    validation_rules_files = list(testpath.rglob(VALIDATION_RULES_FILE))
    dirs = [p.parent for p in validation_rules_files]

    if testpath not in dirs:
        dirs.insert(0, testpath)

    errors = []
    for path in dirs:
        rules = {}
        rules[EXCLUDE] = [str(d) for d in dirs if d != path and not path.is_relative_to(d)]
        inherited_rules = inherit_rules(testpath, path, force)
        rules = combine_rules(combine_rules(rules, testpath_rules),
                              combine_rules(inherited_rules, load_rules(path, force)))
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
    force = [r.strip() for r in args.force.split(",") if not r.isspace()]

    success = test(root_directory, input_directory, force)

    if not success:
        sys.exit(1)
