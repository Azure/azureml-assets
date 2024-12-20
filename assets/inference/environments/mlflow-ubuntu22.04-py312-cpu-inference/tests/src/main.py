# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Validate minimal inference cpu environment by running azmlinfsrv."""

# imports
import json
import os
import subprocess
import requests
from datetime import datetime, timedelta
import time
import argparse


def main(args):
    """Start inference server and post scoring request."""
    # start the server
    server_process = start_server("/var/tmp",
                                  ["--entry_script", args.score, "--port", "8081"],
                                  args.model_dir)

    # score a request
    with open(args.score_input) as f:
        payload_data = json.load(f)

    headers = {"Content-Type": "application/json"}
    res = score_with_post(headers=headers, data=payload_data)
    server_process.kill()

    print_file_contents("/var/tmp", "stderr.txt")
    print_file_contents("/var/tmp", "stdout.txt")
    print(res)


def start_server(log_directory, args, model_dir, timeout=timedelta(seconds=60)):
    """Start inference server with options."""
    stderr_file = open(os.path.join(log_directory, "stderr.txt"), "w")
    stdout_file = open(os.path.join(log_directory, "stdout.txt"), "w")

    env = os.environ.copy()
    env["AZUREML_MODEL_DIR"] = os.path.dirname(os.path.abspath(__file__))
    env["MLFLOW_MODEL_FOLDER"] = model_dir
    print(os.path.abspath(__file__))
    server_process = subprocess.Popen(["azmlinfsrv"] + args, stdout=stdout_file, stderr=stderr_file, env=env)

    max_time = datetime.now() + timeout

    while datetime.now() < max_time:
        time.sleep(0.25)
        req = None
        try:
            req = requests.get("http://127.0.0.1:8081", timeout=10)
        except Exception as e:
            print(e)

        if req is not None and req.ok:
            break

        # Ensure the server is still running
        status = server_process.poll()
        if status is not None:
            break

    return server_process


def score_with_post(headers=None, data=None):
    """Post scoring request to the server."""
    url = "http://127.0.0.1:8081/score"
    return requests.post(url=url, headers=headers, data=data)


def print_file_contents(log_directory, file_name):
    """Print out file contents."""
    print(log_directory, file_name)
    file_path = os.path.join(log_directory, file_name)
    try:
        with open(file_path, 'r') as file:
            contents = file.read()
            print(contents)
    except FileNotFoundError:
        print("file path is not valid.")


def parse_args():
    """Parse input arguments."""
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--score", type=str)
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--score_input", type=str)

    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()

    # run main function
    main(args)
