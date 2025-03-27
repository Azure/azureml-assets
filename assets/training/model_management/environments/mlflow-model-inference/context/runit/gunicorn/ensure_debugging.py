# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Module to run debug processes."""

# This script is added to ensure that local debugging for online endpoints
# (https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-managed-online-endpoints-visual-studio-code)
# continue to work after we switch to launching the inference server with azmlinfsrv.
# Local debugging works by injecting
# some code to /var/azureml-server/entry.py to launch debugpy before the server is started:
# As azmlinfsrv doesn't use /var/azureml-server/entry.py, we have to do the same to the entry.py used by azmlinfsrv.
# While entry.py is updated with this logic in version 0.7.2, this script makes sure local debugging also works for
# servers before version 0.7.2. We should remove this script after most users are on version 0.7.2 or after.

import os
import sys
import textwrap


try:
    import azureml_inference_server_http
except ModuleNotFoundError:
    print("The azureml_inference_server_http package is not installed.")
    sys.exit(1)


if getattr(azureml_inference_server_http, "_HAS_DEBUGPY_SUPPORT", False):
    print("azureml_inference_server_http has support for debugpy. Patching is not necessary.")
    sys.exit(0)


# Patch entry.py to launch debugpy at start.
package_dir = os.path.dirname(azureml_inference_server_http.__file__)
entry_script_path = os.path.join(package_dir, "server", "entry.py")
try:
    with open(entry_script_path, mode="r") as fp:
        code = fp.read()

    # Since the run script can be executed multiple times, make sure we only patch entry.py once.
    if "debugpy" in code:
        print(f"debugpy is already injected into entry.py at {entry_script_path}. Not patching again.")
    else:
        with open(entry_script_path, mode="w", encoding="utf-8") as fp:
            fp.write(
                textwrap.dedent(
                    """\
                    import os
                    import debugpy
                    debugpy.connect(int(os.environ["AZUREML_DEBUG_PORT"]))
                    debugpy.wait_for_client()

                    """
                )
                + code
            )

        print(f"Injected debugpy into entry.py at {entry_script_path}")

except Exception:
    print(f"Failed to inject debugpy into {entry_script_path}")
    raise
