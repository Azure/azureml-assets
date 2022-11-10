# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# imports
import os
import subprocess
import requests
from datetime import datetime, timedelta
import time
import argparse


# define functions
def main(args):
    # start the server
    server_process = start_server("/var/tmp", ["--entry_script", args.score])
    
    # score a request
    req = score_with_post()
    server_process.kill()

    print(req)


def start_server(log_directory, args, timeout=timedelta(seconds=15)):

    stderr_file = open(os.path.join(log_directory, "stderr.txt"), "w")
    stdout_file = open(os.path.join(log_directory, "stdout.txt"), "w")

    env = os.environ.copy()
    server_process = subprocess.Popen(["azmlinfsrv"] + args, stdout=stdout_file, stderr=stderr_file, env=env)

    max_time = datetime.now() + timeout

    while datetime.now() < max_time:
        time.sleep(0.25)
        req = None
        try:
            req = requests.get(f"http://127.0.0.1:5001", timeout=10)
        except Exception as e:
            print(e)

        if req != None and req.ok:
            break

        # Ensure the server is still running
        status = server_process.poll()
        if status is not None:
            break

    print(log_directory, "stderr.txt")
    print(log_directory, "stdout.txt")

    return server_process


def score_with_post(headers=None, data=None):
    url = f"http://127.0.0.1:5001/score"
    return requests.post(url=url, headers=headers, data=data)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--score", type=str)

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
