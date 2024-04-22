# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preset."""

from enum import Enum

from httpx import Timeout


class Preset(Enum):
    """Preset."""

    MTEB_MAIN_EN = "mteb_main_en"


class DeploymentType(Enum):
    """Deployment Type."""

    AOAI = "AOAI"
    OAI = "OAI"
    OSS_MaaS = "OSS_MaaS"
    OSS_MaaP = "OSS_MaaP"


class EmbeddingConstants:
    """Constants for embedding benchmarking."""

    DEFAULT_HTTPX_TIMEOUT = Timeout(timeout=600.0, connect=120.0)
