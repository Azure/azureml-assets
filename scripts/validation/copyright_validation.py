# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Ensure copyright header is at the top of Python scripts."""

import argparse
import sys
from pathlib import Path
from typing import List

COPYRIGHT = [
    "# Copyright (c) Microsoft Corporation.",
    "# Licensed under the MIT License."
]


def _test(testpaths: List[Path], excludes: List[Path] = []) -> bool:
    badfiles = []
    for testpath in testpaths:
        for file in testpath.rglob("*.py"):
            # Skip ignored files
            if file.parent in excludes:
                continue

            # Ignore zero-length files
            if file.stat().st_size == 0:
                continue

            # Read copyright
            with open(file, encoding="utf8") as f:
                for i in range(0, len(COPYRIGHT)):
                    if f.readline().rstrip() != COPYRIGHT[i]:
                        badfiles.append(file)
                        break

    if len(badfiles) > 0:
        print("File(s) missing copyright header:")
        for line in badfiles:
            print(line)
        return False
    return True


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-directories", required=True, type=Path, nargs='+',
                        help="Directories to validate")
    parser.add_argument("-e", "--excludes", default=[], type=Path, nargs='+',
                        help="Directories to exclude")
    args = parser.parse_args()

    success = _test(args.input_directories, args.excludes)

    if not success:
        sys.exit(1)
