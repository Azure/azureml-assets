# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import sys
from pathlib import Path

COPYRIGHT = [
    "# ---------------------------------------------------------",
    "# Copyright (c) Microsoft Corporation. All rights reserved.",
    "# ---------------------------------------------------------"
]


def test(testpath: Path) -> bool:
    badfiles = []
    ignore = [Path(p) for p in []]

    for file in testpath.rglob("*.py"):
        # Skip ignored files
        if file.parent in ignore:
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
    parser.add_argument("-i", "--input-directory", required=True, type=Path, help="Directory to validate")
    args = parser.parse_args()

    success = test(args.input_directory)

    if not success:
        sys.exit(1)
