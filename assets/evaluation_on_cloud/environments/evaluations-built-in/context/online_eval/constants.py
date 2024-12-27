# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Constants for the online evaluation context."""


class TelemetryConstants:
    APP_INSIGHT_HANDLER_NAME = "AppInsightsHandler"
    NON_PII_MESSAGE = '[Hidden as it may contain PII]'


class ExceptionTypes:
    """AzureML Exception Types."""

    User = "User"
    System = "System"
    Service = "Service"
    Unclassified = "Unclassified"
    All = {User, System, Service, Unclassified}


class Queries:
    """Queries for the online evaluation context."""

    LAST_EXECUTION_TIME_QUERY = """
    traces
| where message contains "Schedule ID: {schedule_id} | Last Run Time:"
| order by timestamp desc
| limit 1
| project message"""
