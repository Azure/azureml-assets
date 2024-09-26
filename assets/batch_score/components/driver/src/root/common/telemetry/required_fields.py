# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Defines Part A of the logging schema, keys that have a common meaning across telemetry data.

The schema is defined in: https://msdata.visualstudio.com/Vienna/_wiki/wikis/Vienna.wiki/4672/Common-Schema.
"""
from datetime import datetime, timezone
import uuid


class RequiredFields:
    """Defines Part A of the logging schema, keys that have a common meaning across telemetry data."""

    def __init__(self, subscriptionId, workspaceId, correlationId, componentName, eventName) -> None:
        """Initialize a new instance of the RequiredFields."""
        # pylint: disable=invalid-name
        self.EventId = str(uuid.uuid4())
        self.SubscriptionId = subscriptionId
        self.WorkspaceId = workspaceId
        self.CorrelationId = correlationId
        self.ComponentName = componentName
        self.EventName = eventName
        self.EventTime = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(tz=None).isoformat()
