# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Routing simulator."""

import os
import re

from service_simulator import ServiceSimulator


# If this pattern appears in the environment or arguments, then this simulator will be enabled.
ARG_HOST_PATTERN = "{{ROUTING_SIMULATOR_HOST}}"

ENV_ENDPOINT_URI = "ROUTING_SIMULATOR_ENDPOINT_URI"


class RoutingSimulator(ServiceSimulator):
    """Routing simulator."""

    def __init__(self):
        """Init function."""
        super().__init__(handler=self.RequestHandler)

    @classmethod
    def initialize(cls):
        """Initialize function."""
        if not cls.arg_pattern_is_present(ARG_HOST_PATTERN) or not os.environ.get(ENV_ENDPOINT_URI):
            print("Not starting routing simulator")
            return

        print("Starting routing simulator")

        quota = cls()
        quota.start()

        cls.arg_pattern_replace(ARG_HOST_PATTERN, quota.host)

    class RequestHandler(ServiceSimulator.RequestHandler):
        """Simulator request handler."""

        def do_POST(self):
            """Simulate a POST request."""
            try:
                path = self.path.split("?")[0]

                if path.endswith("/listEndpoints"):
                    service_namespace, endpoint_pool = re.search(r"/serviceNamespaces/([^/]+)/endpointPools/([^/]+)",
                                                                 path,
                                                                 re.IGNORECASE).groups()

                    self.send_json_response({
                        "serviceNamespace": service_namespace,
                        "quotaScope": f"endpointPools:{endpoint_pool}:trafficGroups:batch",
                        "trafficGroup": "batch",
                        "endpoints": [
                            {
                                "endpointUri": os.environ[ENV_ENDPOINT_URI],
                                "trafficWeight": 100,
                            }
                        ]
                    })
                elif path.endswith("/listRoutes"):
                    self.send_json_response({
                        "routes": [
                            {
                                "endpointUri": os.environ[ENV_ENDPOINT_URI],
                                "trafficGroup": "batch",
                                "trafficWeight": 100,
                            }
                        ]
                    })
                else:
                    self.send_error(404)
            except Exception as e:
                self.send_error(500, explain=str(e))
                raise
