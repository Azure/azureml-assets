# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import time
import subprocess
import os
import requests


def call_with_output(command):
    success = False
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT).decode()
        success = True
    except subprocess.CalledProcessError as e:
        output = e.output.decode()
    except Exception as e:
        # check_call can raise other exceptions, such as FileNotFoundError
        output = str(e)
    return (success, output)


class EndpointUtil:
    def __init__(self, model_path, wrapped_model_path):
        self.model_path = model_path
        self.wrapped_model_path = wrapped_model_path
        self.inference_conda_env = "mlflow_inference"
        self.serv_address = "http://127.0.0.1:5000"

    def create_conda_environment(self):
        command = ["apt", "install", "libopenmpi-dev", "--yes"]
        success, output = call_with_output(command)

        command = [
            "conda",
            "env",
            "create",
            "--name",
            self.inference_conda_env,
            "--file",
            os.path.join(self.model_path, "conda.yaml"),
        ]
        success, output = call_with_output(command)

        return success, output

    def create_wrapped_model(self):
        command = [
            "conda",
            "run",
            "-n",
            self.inference_conda_env,
            "python",
            "./wrapmodel.py",
            "--model_path",
            self.model_path,
            "--wrapped_model_path",
            self.wrapped_model_path,
        ]
        success, output = call_with_output(command)
        return success, output

    def create_endpoint(self):
        create_endpoint = subprocess.Popen(
            args=[
                "conda",
                "run",
                "--live-stream",
                "-n",
                self.inference_conda_env,
                "mlflow",
                "models",
                "serve",
                "-m",
                self.wrapped_model_path,
                "--env-manager",
                "local",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            cwd=os.path.dirname(os.path.realpath(__file__)),
        )

        # try for 3 minutes in 10 seconds interval
        retry_max = 18
        retry_wait = 10
        retry_count = 0

        while True:
            try:
                if create_endpoint.returncode is not None:
                    # Server has crashed
                    raise RuntimeError("MLFlow server has crashed")

                requests.get(self.serv_address)
            except requests.ConnectionError:
                retry_count += 1
                if retry_count >= retry_max:
                    raise RuntimeError(
                        "MLFlow server not available after {} seconds.".format(
                            retry_max * retry_wait
                        )
                    )
                time.sleep(retry_wait)
                continue

            return True, self.serv_address
