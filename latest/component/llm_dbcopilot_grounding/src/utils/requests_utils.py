# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Requests utils."""
import logging
import traceback

import requests
import retrying


@retrying.retry(
    wait_fixed=1000,
    stop_max_attempt_number=3,
    retry_on_exception=lambda e: isinstance(e, requests.exceptions.HTTPError)
    and e.response.status_code >= 500,
)
def request(method, url, json=None, headers=None):
    """Request with retry."""
    # if headers is None:
    #     headers = self.default_header
    if json is None:
        json = {}
    response = requests.request(method, url, json=json, headers=headers)
    if response.status_code >= 300 or response.status_code < 200:
        logging.error(f"request failed: {response.status_code}, {response.content}")
        logging.error(f"details: {traceback.format_exc()}")
        response.raise_for_status()
    logging.info(f"response: {response.status_code}, {response.content}")
    return response
