# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch Reference"""

from dataclasses import dataclass


@dataclass
class BatchReference:
    """Details of the Batch Job"""

    aoai_account_name: str
    batch_id: str
    resource_group_name: str
    subscription_id: str
