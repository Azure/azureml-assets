# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Component base class."""
import argparse
import functools
import inspect
import json
import logging
import os
import sys
import time
from functools import cached_property
from typing import Dict, Optional

from azure.ai.ml import MLClient
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import DefaultAzureCredential, ManagedIdentityCredential
from azureml.core import Run, Workspace
from azureml.core.authentication import AzureCliAuthentication, MsiAuthentication
from db_copilot_tool.telemetry import set_print_logger, track_function


def main_entry_point(function_name: str):
    """main_entry_point."""

    def decorator(cls):
        @track_function(name=f"{cls.__name__}.{function_name}")
        def wrapper(**kwargs):
            instance = cls()
            getattr(instance, function_name)(**kwargs)

        set_print_logger()
        if len(sys.argv) > 0:
            class_file_name = os.path.abspath(inspect.getsourcefile(cls))
            entry_file_name = os.path.abspath(sys.argv[0])
            if os.path.basename(entry_file_name) == os.path.basename(class_file_name):
                logging.basicConfig(
                    format="[%(asctime)s] [%(levelname)s] [%(filename)s] %(message)s",
                    level=logging.INFO,
                )
                logging.info("Arguments: %s", sys.argv)
                parser = argparse.ArgumentParser(
                    description=f"{function_name} command line interface"
                )
                sig = inspect.signature(getattr(cls, function_name))
                for param in sig.parameters.values():
                    if param.name == "self":
                        continue
                    type_value = (
                        param.annotation
                        if param.annotation != bool
                        else lambda x: x in ("True", "true", "1", True)
                    )
                    if param.default == inspect.Parameter.empty:
                        parser.add_argument(
                            f"--{param.name}", type=type_value, required=True
                        )
                    else:
                        parser.add_argument(
                            f"--{param.name}", type=type_value, default=param.default
                        )

                # Parse the command-line arguments
                args = parser.parse_args()
                logging.info("\n".join(f"{k}={v}" for k, v in vars(args).items()))
                try:
                    wrapper(**vars(args))
                except Exception as ex:
                    raise ex
                finally:
                    logging.info("finally")
                    time.sleep(5)
        return cls

    return decorator


class ComponentBase:
    """ComponentBase."""

    def __init__(self) -> None:
        """__init__."""
        pass

    parameter_type_mapping = {}

    parameter_mode_mapping = {}

    @classmethod
    def get_parameter_type_mapping(cls) -> Dict[str, str]:
        """get_parameter_type_mapping."""
        return cls.parameter_type_mapping

    @classmethod
    def get_parameter_mode_mapping(cls) -> Dict[str, str]:
        """get_parameter_mode_mapping."""
        return cls.parameter_mode_mapping

    @cached_property
    def workspace(self):
        """workspace."""
        return self.run.experiment.workspace

    @functools.cached_property
    def run(self) -> Run:
        """run."""
        try:
            run = Run.get_context()
            return run
        except Exception as ex:
            logging.error("Failed to get run from run context. Error: %s", ex)
            raise ex

    @functools.cached_property
    def default_headers(self):
        """default_headers."""
        headers = {
            "Authorization": "Bearer %s" % self.run_token,
            "Content-Type": "application/json",
        }
        return headers

    @functools.cached_property
    def run_token(self):
        """run_token."""
        return os.environ.get("AZUREML_RUN_TOKEN", "")

    @functools.cached_property
    def service_endpoint(self):
        """service_endpoint."""
        return os.environ.get("AZUREML_SERVICE_ENDPOINT", "")

    @functools.cached_property
    def workspace_scope(self):
        """workspace_scope."""
        return os.environ.get("AZUREML_WORKSPACE_SCOPE", "")

    @staticmethod
    def parse_llm_config(config_str: str) -> Optional[str]:
        """parse_llm_config."""
        config = json.loads(config_str)
        return config["deployment_name"]


class OBOComponentBase(ComponentBase):
    """OBOComponentBase."""

    def __init__(self) -> None:
        """__init__."""
        super().__init__()

    @functools.cached_property
    def azureml_auth(self):
        """azureml_auth."""
        client_id = os.environ.get("DEFAULT_IDENTITY_CLIENT_ID", None)
        if "OBO_ENDPOINT" in os.environ:
            logging.info("Using OBO authentication")
            os.environ["MSI_ENDPOINT"] = os.environ.get("OBO_ENDPOINT", "")
            auth = MsiAuthentication()
        elif client_id is not None:
            logging.info("Using Managed Identity authentication")
            auth = ManagedIdentityCredential(client_id=client_id)
        else:
            logging.info("Using Azure CLI authentication")
            auth = AzureCliAuthentication()
        return auth

    @functools.cached_property
    def mlclient_credential(self):
        """mlclient_credential."""
        for key, value in os.environ.items():
            logging.debug("%s: %s", key, value)
        if "OBO_ENDPOINT" in os.environ:
            logging.info("Using OBO authentication")
            credential = AzureMLOnBehalfOfCredential()
            os.environ["MSI_ENDPOINT"] = os.environ.get("OBO_ENDPOINT", "")
        elif "MSI_ENDPOINT" in os.environ and os.environ.get("MSI_ENDPOINT") != "":
            logging.info("Using MSI authentication")
            credential = ManagedIdentityCredential(
                client_id=os.environ.get("DEFAULT_IDENTITY_CLIENT_ID")
            )
        else:
            credential = DefaultAzureCredential(
                exclude_managed_identity_credential=True
            )
        return credential

    @functools.cached_property
    def workspace(self) -> Workspace:
        """workspace."""
        run = Run.get_context()
        workspace: Workspace = run.experiment.workspace
        return workspace

    @functools.cached_property
    def ml_client(self) -> MLClient:
        """ml_client."""
        credential = self.mlclient_credential
        workspace = self.workspace
        logging.info("workspace: %s", workspace)
        client = MLClient(
            credential=credential,
            subscription_id=workspace._subscription_id,
            resource_group_name=workspace._resource_group,
            workspace_name=workspace._workspace_name,
        )
        return client
