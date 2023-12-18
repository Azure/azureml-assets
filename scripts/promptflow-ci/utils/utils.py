# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utils."""

import subprocess


def run_command(command: str):
    """Run cmd."""
    result = subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result


def get_diff_files():
    """Run git diff."""
    run_command('git fetch origin')
    diff_files = run_command('git diff --name-status remotes/origin/main ./')
    return diff_files.stdout.split('\n')