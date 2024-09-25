# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Endpoint simulator."""

import json
import os
import random
import time

from service_simulator import ServiceSimulator


# If this pattern appears in the environment or arguments, then this simulator will be enabled.
ARG_HOST_PATTERN = "{{ENDPOINT_SIMULATOR_HOST}}"

WORK_SECONDS = float(os.environ.get("ENDPOINT_SIMULATOR_WORK_SECONDS") or 10)


class EndpointSimulator(ServiceSimulator):
    """Endpoint simulator."""

    def __init__(self):
        """Init function."""
        super().__init__(handler=self.RequestHandler)

    @classmethod
    def initialize(cls):
        """Initialize function."""
        if not cls.arg_pattern_is_present(ARG_HOST_PATTERN):
            print("Not starting endpoint simulator")
            return

        print("Starting endpoint simulator")

        endpoint = cls()
        endpoint.start()

        cls.arg_pattern_replace(ARG_HOST_PATTERN, endpoint.host)

    class RequestHandler(ServiceSimulator.RequestHandler):
        """Simulator request handler."""

        def do_POST(self):
            """Simulate a POST request."""
            try:
                path = self.path.split("?")[0]

                if path.endswith("/completions"):
                    content_length = int(self.headers.get("Content-Length", -1))
                    request = json.loads(self.rfile.read(content_length))

                    time.sleep(WORK_SECONDS)

                    response = self.__build_response(request)
                    self.send_json_response(response)
                else:
                    self.send_error(404)
            except Exception as e:
                self.send_error(500, explain=str(e))
                raise

        def __build_response(self, request):
            prompt = request["prompt"]
            max_tokens = request.get("max_tokens", 10)

            completion_tokens = random.randint(0, max_tokens)
            prompt_tokens = len(prompt.split())

            completion = "".join([" hello"] * completion_tokens)

            return {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "text": completion
                    }
                ],
                "usage": {
                    "completion_tokens": completion_tokens,
                    "prompt_tokens": prompt_tokens,
                    "total_tokens": completion_tokens + prompt_tokens,
                }
            }
