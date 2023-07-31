# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""General score script for db_copilot_mir."""
import logging
import socket
import subprocess
import threading
import time

import requests
from app_management import AppConfig, start_flask_app
from azureml.contrib.services.aml_request import AMLRequest, rawhttp
from azureml.contrib.services.aml_response import AMLResponse
from db_copilot_tool.telemetry import set_print_logger
from flask import Response, stream_with_context


def pick_random_port():
    """Pick a random port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def run_command(command):
    """Run a command."""
    logging.info(f"Running command: {command}")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    if process:
        while True:
            output = process.stdout.readline()
            error = process.stderr.readline()
            if output == b"" and error == b"" and process.poll() is not None:
                break
            if output:
                logging.info(output.decode().strip())
            if error:
                logging.error(error.decode().strip())
    else:
        logging.error("Failed to start process")


def init():
    """Initialize the score script."""
    set_print_logger()
    global app_thread
    global app_port
    # global rslex_thread
    # rslex_thread = None
    app_thread = None
    app_port = pick_random_port()
    app_config = AppConfig.load_from_file()
    app_config.port = app_port

    def start_flask():
        start_flask_app(app_config)

    if not app_thread or not app_thread.is_alive():
        app_thread = threading.Thread(target=start_flask)
        app_thread.start()

    # rslex_mount = os.environ.get("RSLEX-MOUNT", None)

    # if rslex_mount and (not rslex_thread or not rslex_thread.is_alive()):

    #     def start_rslex():
    #         cmd = f"/tmp/dbcopilot/rslex-fuse-cli --mount-point /tmp/workspace_mount/ --source-url {rslex_mount}"
    #         run_command(cmd)

    #     rslex_thread = threading.Thread(target=start_rslex)
    #     rslex_thread.start()

    time.sleep(3)
    if not app_thread.is_alive():
        raise Exception("App thread failed to start")
    # if rslex_thread and not rslex_thread.is_alive():
    #     raise Exception("Rslex thread failed to start")

    # for file in os.listdir("/tmp/workspace_mount/"):
    #     logging.info(f"File: {file}")
    logging.info(f"Init complete: Pork:{app_port}")


@rawhttp
def run(request: AMLRequest):
    """Score script."""
    logging.info(f"Request: {request.__dict__}")
    api = request.headers.get("api", None)
    is_stream = request.headers.get("is_stream", False)
    if is_stream:
        is_stream = is_stream in ("true", "True")
    if api:
        api.lstrip("/")
    url = f"http://localhost:{app_port}/{api}" if api else f"http://localhost:{app_port}"
    logging.info(f"Router url: {url}. Is stream: {is_stream}")
    response = requests.request(request.method, url, headers=request.headers, data=request.data, stream=is_stream)
    logging.info("Request processed")
    if is_stream:

        def generate():
            for content in response.iter_content(chunk_size=1024):
                logging.info(f"Content: {content}")
                yield content
            logging.info("Stream complete")

        return Response(stream_with_context(generate()), mimetype="application/json", status=200)
    else:
        return AMLResponse(response.text, response.status_code)