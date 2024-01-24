# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# flake8: noqa: F401

"""__init__."""

import subprocess
import sys

# TODO: remove this hack after dedicated environment is published.
subprocess.check_call([sys.executable,"-m", "pip", "install",
                       'PyDispatcher==2.0.7',
                       'StrEnum==0.4.15',
                       'tiktoken==0.5.2'])
