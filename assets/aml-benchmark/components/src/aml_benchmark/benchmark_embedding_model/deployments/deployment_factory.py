# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Deployment Factory."""

from typing import Optional

from azureml._common._error_definition.azureml_error import AzureMLError

from .aoai_deployment import AOAIDeployment
from .oai_deployment import OAIDeployment
from .oss_deployment import OSSDeployment
from ..utils.constants import DeploymentType
from ...utils.helper import get_api_key_from_connection
from ...utils.exceptions import BenchmarkValidationException
from ...utils.error_definitions import BenchmarkValidationError


class DeploymentFactory:
    """Factory for instantiating deployments."""

    @staticmethod
    def get_deployment(
        deployment_type: str,
        deployment_name: str,
        endpoint_url: Optional[str],
        connections_name: Optional[str],
    ):
        """Get deployment instance."""
        _api_key, _api_version = get_api_key_from_connection(connections_name)

        if deployment_type == DeploymentType.AOAI.value:
            return AOAIDeployment(
                deployment_name=deployment_name,
                endpoint_url=endpoint_url,
                api_key=_api_key,
                api_version=_api_version,
            )
        elif deployment_type == DeploymentType.OAI.value:
            return OAIDeployment(
                deployment_name=deployment_name,
                api_key=_api_key,
            )
        elif deployment_type == DeploymentType.OSS.value:
            return OSSDeployment(
                deployment_name=deployment_name,
                endpoint_url=endpoint_url,
                api_key=_api_key,
            )
        else:
            mssg = "Unknown deployment type. Choose from one of the deployment types: `AOAI`, `OAI`, `OSS`."
            raise BenchmarkValidationException._with_error(
                AzureMLError.create(BenchmarkValidationError, error_details=mssg)
            )
        
        