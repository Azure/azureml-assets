# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains utilities to format or parse a datetime object."""

from datetime import datetime
from dateutil import parser


def parse_datetime_from_string(format: str, date_to_format: str) -> datetime:
    """Parse a datetime from a string according the format."""
    parsed_date = parser.parse(date_to_format)
    return datetime.strptime(
        str(parsed_date.strftime(format)), format
    )
