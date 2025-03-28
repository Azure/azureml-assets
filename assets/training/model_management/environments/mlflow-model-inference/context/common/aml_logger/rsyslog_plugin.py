# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module for system log."""

import sys
import recorder_factory


def log_stream():
    """Log message stream."""
    while True:
        raw_msg = sys.stdin.readline()
        if raw_msg:
            yield raw_msg.rstrip()
        else:
            break


if __name__ == "__main__":
    log_recorder = recorder_factory.get_recorders()
    for raw_msg in log_stream():
        log_recorder.on_receive(raw_msg)
        sys.stdout.flush()
    log_recorder.on_exit()
