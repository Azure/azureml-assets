# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Class for Claudie online endpoint."""
from typing import Dict, Optional, Tuple

import hashlib
import hmac
import os

from datetime import datetime, UTC

from aml_benchmark.utils.online_endpoint.online_endpoint import OnlineEndpoint
from aml_benchmark.utils.online_endpoint.online_endpoint_model import OnlineEndpointModel
from aml_benchmark.batch_benchmark_score.batch_score.utils.exceptions import BenchmarkUserException
from aml_benchmark.batch_benchmark_score.batch_score.utils.error_definitions import BenchmarkUserError
from azureml._common._error_definition.azureml_error import AzureMLError


class ClaudieOnlineEndpoint(OnlineEndpoint):
    """
    The class for Claudie Online endpoint.

    **Note:** Please see last three parameters in the list, the rest is not relevant
              to external endpoints.
    Plaese provide AccessKey and SecretKey as an environmental variables.
    For reference see https://docs.aws.amazon.com/IAM/latest/UserGuide/create-signed-request.html
    :param workspace_name: The name of the workspace
                           (may be any string, it is not relevant to external endpoints)
    :param resource_group: The name of the resource group
                           (may be any string, it is not relevant to external endpoints)
    :param subscription_id: The Azure subscription ID.
                           (may be any string, it is not relevant to external endpoints)
    :param online_endpoint_url: Endpoint url (not used)
    :param endpoint_name: Endpoint name (not used)
    :param deployment_name: The deployment name (not used)
    :param sku: Compute sku (not used)
    :param online_endpoint_model: Online endpoint model (not used)
    :param connections_name: Connection name (not used)
    :param aws_region: The region, where the endpoint is located, for example, 'us-east-1'
    :param model_identifier: The model to be used, for example 'anthropic.claude-v2'
    :param payload: The request payload i.e. prompt. To get Authentication header it needs to be hashed.
    """

    ACCESS_KEY = "AccessKey"
    SECRET_KEY = "SecretKey"
    # Private static variables
    _SERVICE = 'bedrock'

    def __init__(
        self,
        workspace_name: str,
        resource_group: str,
        subscription_id: str,
        online_endpoint_url: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        deployment_name: Optional[str] = None,
        sku: Optional[str] = None,
        online_endpoint_model: Optional[OnlineEndpointModel] = None,
        connections_name: Optional[str] = None,
        aws_region: str = None,
        model_identifier: str = None,
        payload: str = None,
    ):
        """Constructor."""
        super().__init__(
            workspace_name,
            resource_group,
            subscription_id,
            online_endpoint_url,
            endpoint_name,
            deployment_name,
            sku,
            online_endpoint_model,
            connections_name
        )
        for name, param in (
            ('aws_region', aws_region),
            ('model_identifier', model_identifier),
                ('payload', payload)):
            if not param:
                raise BenchmarkUserException._with_error(
                    AzureMLError.create(
                        BenchmarkUserError,
                        error_details=(f"Please provide the {name} parameter."))
                )
        self._aws_region = aws_region
        self._model_identifier = model_identifier
        self._payload_sha_256 = ClaudieOnlineEndpoint._sha256_sum(payload)
        self._access_key = os.environ.get(ClaudieOnlineEndpoint.ACCESS_KEY)
        self._secret_key = os.environ.get(ClaudieOnlineEndpoint.SECRET_KEY)
        if not self._access_key or not self._secret_key:
            raise BenchmarkUserException._with_error(
                AzureMLError.create(
                    BenchmarkUserError,
                    error_details=("AccessKey or SecretKey are empty"
                                   f"Please provide AccessKey in {ClaudieOnlineEndpoint.ACCESS_KEY} "
                                   f"and SecretKey in {ClaudieOnlineEndpoint.SECRET_KEY} environmental variables."))
            )
        self._aws_region = aws_region

    @staticmethod
    def _sha256_sum(data: str) -> str:
        """
        Generate the hexdigest for data.

        :param data: The data to generate digest for.
        :return: The hexadecimal sha256 digest of a string.
        """
        sha = hashlib.sha256()
        sha.update(data.encode('utf-8'))
        return sha.hexdigest()

    @property
    def payload_hash(self) -> str:
        """Return the payload Sha256 hash."""
        return self._payload_sha_256

    @property
    def scoring_url(self) -> str:
        """Return the scoring URI for the Claudie endpoint."""
        return f"https://bedrock-runtime.{self._aws_region}.amazonaws.com/model/{self._model_identifier}/invoke"

    def _get_canonical_headers(self, timestamp: str) -> Dict[str, str]:
        """
        Return the canonical headers.

        :return: The dictionary with the canonical headers.
        """
        return {
            'accept': 'application/json',
            'host': 'bedrock-runtime.us-east-1.amazonaws.com',
            'content-type': 'application/json',
            'X-Amz-Content-Sha256': self.payload_hash,
            'X-Amz-Date': timestamp
        }

    def _get_canonical_header_string_and_signed_headers(self, headers) -> Tuple[str, str]:
        """
        Return the tuple with canonical headers as a string and signing headers.

        :param headers: canonical headers.
        :return: canonical headers and signing headers.
        """
        formatted_headers = {k.lower(): v.strip() for k, v in headers.items()}
        canonic_headers = sorted(formatted_headers.keys())
        canonical_headers_str = "\n".join([f'{k}:{formatted_headers[k]}' for k in canonic_headers]) + '\n'
        return canonical_headers_str, ";".join([k for k in canonic_headers])

    def _get_signing_key(self, timestamp: str) -> bytes:
        """
        Return the signing key, to create a Authentication header.

        :param timestamp: the timestemp in ISO 8601 format.
        :returns: The bytes, used to sign the request.
        """
        date_key = hmac.new(
            f"AWS4{self._secret_key}".encode('utf-8'), timestamp[:8].encode('utf-8'), hashlib.sha256).digest()
        date_region_key = hmac.new(
            date_key, self._aws_region.encode('utf-8'), hashlib.sha256).digest()
        date_region_service_key = hmac.new(
            date_region_key, ClaudieOnlineEndpoint._SERVICE.encode('utf-8'),
            hashlib.sha256).digest()
        return hmac.new(
            date_region_service_key, "aws4_request".encode('utf-8'), hashlib.sha256).digest()

    def _get_date_and_time(self) -> Tuple[str, str]:
        """
        Return date and time in ISO 8601 format.

        **Note:** This method is modtly done for ability to test header generation.
        :return: The tuple with datetime and date.
        """
        date_time = datetime.now(tz=UTC).strftime('%Y%m%dT%H%M%SZ')
        return date_time, date_time[:8]

    def get_endpoint_authorization_header(self) -> Dict[str, str]:
        """
        Get authentication headers for the Claudie model.

        :return: The dictionary representation of headers.
        """
        date_time, date = self._get_date_and_time()
        headers = self._get_canonical_headers(date_time)
        canonical_headers_str, signed_headers_str = self._get_canonical_header_string_and_signed_headers(headers)
        canonical_request_str = (
            "POST\n"
            "/model/anthropic.claude-v2/invoke\n"
            "\n"   # No query string
            f"{canonical_headers_str}\n"
            f"{signed_headers_str}\n"
            f"{self.payload_hash}"
        )
        hashed_canonocal_request = ClaudieOnlineEndpoint._sha256_sum(canonical_request_str)
        string_to_sign = ("AWS4-HMAC-SHA256\n"
                          f"{date_time}\n"
                          f"{date}/{self._aws_region}/{ClaudieOnlineEndpoint._SERVICE}/aws4_request\n"
                          f"{hashed_canonocal_request}"
                          )
        signing_key = self._get_signing_key(date_time)
        signature = hmac.new(signing_key, string_to_sign.encode('utf-8'), hashlib.sha256).hexdigest()
        # Finally we will add the signature into the headers
        headers['Authorization'] = (
            'AWS4-HMAC-SHA256 Credential='
            f'{self._access_key}/{date}/{self._aws_region}/{ClaudieOnlineEndpoint._SERVICE}/aws4_request, '
            f'SignedHeaders={signed_headers_str}, '
            f'Signature={signature}'
        )
        return headers
