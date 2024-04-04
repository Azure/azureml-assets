# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preset."""

from enum import Enum


class Preset(Enum):
    """Preset."""

    MTEB_MAIN_EN = "mteb_main_en"


class DeploymentType(Enum):
    """Deployment Type."""

    AOAI = "AOAI"
    OAI = "OAI"
    OSS = "OSS"