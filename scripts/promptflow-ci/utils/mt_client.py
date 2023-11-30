# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MTClient class."""

from datetime import datetime

import requests

from .authentication import get_service_principal_authentication_header, \
    get_interactive_login_authentication_header
from .retry_helper import retry_helper
from .logging_utils import log_debug


class MTClient:
    """MTClient class."""
    flow_api_endpoint = "{MTServiceRoute}/api/subscriptions/{SubscriptionId}/resourceGroups/{ResourceGroupName}/" \
                        "providers/Microsoft.MachineLearningServices/workspaces/{WorkspaceName}/flows"
    create_flow_api_format = "{0}/?experimentId={1}"
    create_flow_from_sample_api_format = "{0}/fromsample?experimentId={1}"
    list_flows_api_format = "{0}/?experimentId={1}&ownedOnly={2}&flowType={3}"
    submit_flow_api_format = "{0}/submit?experimentId={1}&endpointName={2}"
    submit_flow_api_without_endpoint_name_format = "{0}/submit?experimentId={1}"
    list_flow_runs_api_format = "{0}/{1}/runs?experimentId={2}&bulkTestId={3}"
    get_flow_run_status_api_format = "{0}/{1}/{2}/status?experimentId={3}"
    list_bulk_tests_api_format = "{0}/{1}/bulkTests"
    get_bulk_tests_api_format = "{0}/{1}/bulkTests/{2}"
    deploy_flow_api_format = "{0}/deploy?asyncCall={1}"
    get_samples_api_format = "{0}/samples"

    def __init__(self, mt_service_route,
                 subscription_id, resource_group_name, workspace_name,
                 tenant_id=None, client_id=None, client_secret=None):
        """MT Client init."""
        self.mt_service_route = mt_service_route
        self.subscription_id = subscription_id
        self.resource_group_name = resource_group_name
        self.workspace_name = workspace_name
        self.api_endpoint = self.flow_api_endpoint.format(
            MTServiceRoute=self.mt_service_route,
            SubscriptionId=self.subscription_id,
            ResourceGroupName=self.resource_group_name,
            WorkspaceName=self.workspace_name)
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

    def _request(self, method, url, **kwargs):
        """MT client request call."""
        if self.tenant_id is None or self.client_id is None or self.client_secret is None:
            # Interactive login authentication for local debugging
            header = get_interactive_login_authentication_header()
        else:
            header = get_service_principal_authentication_header(tenant_id=self.tenant_id, client_id=self.client_id,
                                                                 client_secret=self.client_secret)

        resp = method(url, **{**kwargs, "headers": header})
        if method.__name__ == "post":
            log_debug(
                f"[Request] {method.__name__} API Request id: {header['x-ms-client-request-id']}")
        if resp.status_code != 200:
            raise requests.exceptions.HTTPError(
                f"{method.__name__} on url {url} failed with status code [{resp.status_code}. Error: {resp.json()}].",
                response=resp)
        return resp.json()

    def _get(self, url):
        """Mt client get request."""
        return self._request(requests.get, url)

    def _post(self, url, json_body):
        """Mt client post request."""
        return self._request(requests.post, url, json=json_body)

    @retry_helper()
    def create_or_update_flow(self, json_body):
        """Create flow."""
        url = self.api_endpoint
        result = self._post(url, json_body)
        return result

    @retry_helper()
    def create_flow_from_sample(self, json_body, experiment_id):
        """Create flow from sample json."""
        url = self.create_flow_from_sample_api_format.format(
            self.api_endpoint, experiment_id)
        result = self._post(url, json_body)
        return result

    @retry_helper()
    def submit_flow(self, json_body, experiment_id):
        """Submit flow with a created flow run id or evaluation flow run id"""
        url = self.submit_flow_api_without_endpoint_name_format.format(
            self.api_endpoint, experiment_id)
        # We need to update flow run id in case retry happens, submit same json body with same flow run id will cause
        # 409 error.
        # Update flow run id
        flow_run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        evaluation_run_id = f"evaluate_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        json_body['flowRunId'] = flow_run_id
        # Update evaluation run id for BulkTest run
        if "evaluationFlowRunSettings" in json_body['flowSubmitRunSettings'] and "evaluation" in \
                json_body['flowSubmitRunSettings']['evaluationFlowRunSettings']:
            json_body['flowSubmitRunSettings']['evaluationFlowRunSettings']["evaluation"]["flowRunId"] = \
                evaluation_run_id
        result = self._post(url, json_body)

        return result, flow_run_id, evaluation_run_id

    @retry_helper()
    def get_run_status(self, experiment_id, flow_id, run_id):
        url = self.get_flow_run_status_api_format.format(
            self.api_endpoint, flow_id, run_id, experiment_id)
        result = self._get(url)
        return result


def get_mt_client(
        subscription_id,
        resource_group,
        workspace_name,
        tenant_id,
        client_id,
        client_secret,
        is_local=False,
        mt_service_route="https://eastus2euap.api.azureml.ms/flow") -> MTClient:
    """Get mt client."""
    if (is_local):
        mt_client = MTClient(mt_service_route, subscription_id,
                             resource_group, workspace_name)
    else:
        mt_client = MTClient(mt_service_route, subscription_id, resource_group, workspace_name, tenant_id, client_id,
                             client_secret)
    return mt_client
