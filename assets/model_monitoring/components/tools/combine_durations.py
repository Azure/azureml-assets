# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility file for model-monitoring-ci github workflow."""

import json
import sys
from pathlib import Path


def run():
    """Combine group test_duration files into a single file for model-monitoring-ci."""
    split_prefix = sys.argv[1]
    durations_path = Path(sys.argv[2])

    split_paths = Path(".").glob(f"{split_prefix}*/{durations_path.name}")
    try:
        previous_durations = json.loads(durations_path.read_text())
    except FileNotFoundError:
        previous_durations = {}
    new_durations = previous_durations.copy()

    for path in split_paths:
        durations = json.loads(path.read_text())
        new_durations.update(
            {
                name: duration
                for (name, duration) in durations.items()
                if previous_durations.get(name) != duration
            }
        )

    durations_path.parent.mkdir(parents=True, exist_ok=True)
    durations_path.write_text(json.dumps(new_durations))


if __name__ == "__main__":
    run()
