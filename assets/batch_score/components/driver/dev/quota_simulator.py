# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Quota simulator."""

import json
import os

from datetime import datetime, timedelta, timezone
from functools import partial
from threading import Lock
from uuid import uuid4

from service_simulator import ServiceSimulator


# If this pattern appears in the environment or arguments, then this simulator will be enabled.
ARG_HOST_PATTERN = "{{QUOTA_SIMULATOR_HOST}}"

CAPACITY = int(os.environ.get("QUOTA_SIMULATOR_CAPACITY") or 2048)


class QuotaSimulator(ServiceSimulator):
    """Quota simulator."""

    def __init__(self):
        """Init function."""
        super().__init__(handler=partial(self.RequestHandler, simulator=self))

        self._leases = {}
        self._leases_lock = Lock()

    @classmethod
    def initialize(cls):
        """Initialize function."""
        if not cls.arg_pattern_is_present(ARG_HOST_PATTERN):
            print("Not starting quota simulator")
            return

        print("Starting quota simulator")

        quota = cls()
        quota.start()

        cls.arg_pattern_replace(ARG_HOST_PATTERN, quota.host)

    class RequestHandler(ServiceSimulator.RequestHandler):
        """Simulator request handler."""

        def __init__(self, *args, simulator: 'QuotaSimulator', **kwargs):
            """Init function."""
            self.__simulator = simulator
            super().__init__(*args, **kwargs)

        def do_POST(self):
            """Simulate a POST request."""
            try:
                path = self.path.split("?")[0]

                content_length = int(self.headers.get("Content-Length", -1))
                request = json.loads(self.rfile.read(content_length))

                if path.lower().endswith("/requestlease"):
                    with self.__simulator._leases_lock:
                        requested = request["requestedCapacity"]

                        for lease_id, (expiration, _) in list(self.__simulator._leases.items()):
                            if expiration < datetime.now(timezone.utc):
                                del self.__simulator._leases[lease_id]

                        if sum(c for _, c in self.__simulator._leases.values()) + requested <= CAPACITY:
                            lease_id = str(uuid4())

                            print(f"Quota simulator: Capacity lease {lease_id} granted.")

                            duration = timedelta(seconds=30)
                            expiration = datetime.now(timezone.utc) + duration
                            self.__simulator._leases[lease_id] = expiration, requested

                            self.send_json_response({
                                "leaseId": lease_id,
                                "leaseDuration": str(duration),
                                "leaseExpiryTime": expiration.isoformat(),
                            })
                        else:
                            self.send_json_response({"error": "Not enough capacity to lease."}, code=429)
                elif path.lower().endswith("/renewlease"):
                    with self.__simulator._leases_lock:
                        old_lease_id = request["leaseId"]

                        old_expiration, capacity = self.__simulator._leases[old_lease_id]
                        assert datetime.now(timezone.utc) < old_expiration

                        new_lease_id = str(uuid4())
                        new_duration = timedelta(seconds=10)
                        new_expiration = datetime.now(timezone.utc) + new_duration

                        del self.__simulator._leases[old_lease_id]
                        self.__simulator._leases[new_lease_id] = new_expiration, capacity

                        print(f"Quota simulator: Capacity lease {old_lease_id} renewed as {new_lease_id}.")

                        self.send_json_response({
                            "leaseId": new_lease_id,
                            "leaseDuration": str(new_duration),
                            "leaseExpiryTime": new_expiration.isoformat(),
                        })
                elif path.lower().endswith("/releaselease"):
                    with self.__simulator._leases_lock:
                        lease_id = request["leaseId"]

                        del self.__simulator._leases[lease_id]

                        print(f"Quota simulator: Capacity lease {lease_id} deleted.")

                        self.send_json_response({})
                else:
                    self.send_error(404)
            except Exception as e:
                self.send_error(500, explain=str(e))
                raise
