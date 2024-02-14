# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains StoreUrl class."""

from urllib.parse import urlparse
import os
import re
from typing import Union
from azure.identity import ClientSecretCredential
from azure.core.credentials import AzureSasCredential
from azure.storage.blob import ContainerClient
from azure.storage.filedatalake import FileSystemClient
from azureml.core import Workspace, Run, Datastore
from shared_utilities.momo_exceptions import InvalidInputError


class StoreUrl:
    """Helper class to Convert base_path to HDFS path."""

    def __init__(self, base_url: str, ws: Workspace = None):
        """Initialize a StoreUrl instance with a store url string."""
        self._base_url = base_url
        self._set_properties(ws)

    def get_hdfs_url(self, relative_path: str = None) -> str:
        """
        Get HDFS url for this store url.

        :param relative_path: relative path to the base path
        :return: HDFS url, will be abfs(s) path for gen2 and wasb(s) for blob store
        """
        return self._get_url(relative_path=relative_path)

    def get_abfs_url(self, relative_path: str = None) -> str:
        """
        Get abfs url for the store url.

        :param relative_path: relative path to the base path
        :return: always abfs(s) url, this is very helpful to access append blob in blob store
        """
        scheme = "abfss" if (not self.is_local_path()) and self._is_secure() else "abfs"
        return self._get_url(scheme=scheme, store_type="dfs", relative_path=relative_path)

    def get_azureml_url(self, relative_path: str = None) -> str:
        """
        Get azureml url for the store url.

        :param relative_path: relative path to the base path
        :return: azureml url
        """
        if self._datastore is None:
            raise ValueError(f"{self._base_url} is not an azureml url.")
        url = (f"azureml://subscriptions/{self._datastore.subscription_id}/resourceGroups"
               f"/{self._datastore.resource_group}/workspaces/{self._datastore.workspace.name}"
               f"/datastores/{self._datastore.name}/paths")
        if self.path:
            url = f"{url}/{self.path}"
        if relative_path:
            url = f"{url}/{relative_path.lstrip('/')}"
        return url

    def _get_url(self, scheme=None, store_type=None, relative_path=None) -> str:
        if not self.account_name:
            return f"{self._base_url}/{relative_path}" if relative_path else self._base_url

        scheme = scheme or self._scheme
        store_type = store_type or self.store_type
        url = f"{scheme}://{self.container_name}@{self.account_name}.{store_type}.{self._endpoint}"
        if self.path:
            url = f"{url}/{self.path}"
        if relative_path:
            url = f"{url}/{relative_path.lstrip('/')}"
        return url

    def get_credential(self) -> Union[str, ClientSecretCredential, AzureSasCredential, None]:
        """Get credential for this store url."""
        if not self._datastore:
            raise InvalidInputError(
                "abfs(s), wasb(s) and http(s) URLs are credential less and are NOT supported by Model Monitoring job, "
                "please use azureml file system path with credential data store, e.g. "
                "azureml://datastores/<credential_datastore_name>/paths/<path_to_data>")
        elif self._datastore.datastore_type == "AzureBlob":
            if self._datastore.credential_type == "AccountKey":
                return self._datastore.account_key
            elif self._datastore.credential_type == "Sas":
                return AzureSasCredential(self._datastore.sas_token)
            elif self._datastore.credential_type is None or self._datastore.credential_type == "None":
                raise InvalidInputError("Credential-less input data is NOT supported for Model Monitoring job, "
                                        f"please add credential to datastore {self._datastore.name}.")
            else:
                raise InvalidInputError(f"Unsupported credential type: {self._datastore.credential_type}, "
                                        "only AccountKey and Sas are supported.")
        elif self._datastore.datastore_type == "AzureDataLakeGen2":
            if self._datastore.tenant_id and self._datastore.client_id and self._datastore.client_secret:
                return ClientSecretCredential(tenant_id=self._datastore.tenant_id, client_id=self._datastore.client_id,
                                              client_secret=self._datastore.client_secret)
            else:
                raise InvalidInputError("Credential-less input data is NOT supported for Model Monitoring job. "
                                        f"Please add credential to datastore {self._datastore.name}.")
        else:
            raise InvalidInputError(f"Unsupported datastore type: {self._datastore.datastore_type}, "
                                    "only Azure Blob and Azure Data Lake Gen2 are supported.")

    def get_container_client(self, credential: Union[str, AzureSasCredential, ClientSecretCredential, None] = None) \
            -> Union[FileSystemClient, ContainerClient, None]:
        """
        Get container client for this store url.

        :param credential: if provided, it contains the credential to authorize the container to access the data,
        if not provided, will retrieve credential from datastore. It's a special handling for access dataref file in
        executors.
        """
        if not self.account_name:
            # local or not supported store type
            return None

        # blob, has cred datastore
        if self.store_type == "blob" and self._datastore \
                and (self._datastore.credential_type and self._datastore.credential_type != "None"):
            return self._datastore.blob_service.get_container_client(self.container_name)
        # TODO fallback to DefaultAzureCredential for credential less datastore for now, may need better fallback logic
        credential = credential or self.get_credential()
        account_url_scheme = "https" if self._is_secure() else "http"
        if self.store_type == "blob":
            return ContainerClient(account_url=f"{account_url_scheme}://{self.account_name}.blob.core.windows.net",
                                   container_name=self.container_name, credential=credential)
        elif self.store_type == "dfs":
            return FileSystemClient(account_url=f"{account_url_scheme}://{self.account_name}.dfs.core.windows.net",
                                    file_system_name=self.container_name, credential=credential)
        else:
            raise InvalidInputError(f"Unsupported store type: {self.store_type}, only blob and dfs are supported.")

    def is_folder_exists(self, relative_path: str) -> bool:
        """Check if the folder exists in the store."""
        if self.is_local_path():
            return self._is_local_folder_exists(relative_path)

        container_client = self.get_container_client()
        relative_path = relative_path.strip("/")
        if isinstance(container_client, FileSystemClient):
            return container_client.get_directory_client(f"{self.path}/{relative_path}").exists()
        else:
            full_path = f"{self.path}/{relative_path}/" if relative_path else f"{self.path}/"
            blobs = container_client.list_blobs(name_starts_with=full_path)
            return any(blobs)

    def is_local_path(self) -> bool:
        """Check if the store url is a local path."""
        if not self._base_url:
            return False
        return os.path.isdir(self._base_url) or os.path.isfile(self._base_url) or self._base_url.startswith("file://")\
            or self._base_url.startswith("/") or self._base_url.startswith(".") \
            or re.match(r"^[a-zA-Z]:[/\\]", self._base_url)

    def read_file_content(self, relative_path: str = None,
                          credential: Union[str, AzureSasCredential, ClientSecretCredential, None] = None) -> str:
        """Read file content from the store."""
        if self.is_local_path():
            return self._read_local_file_content(relative_path)

        container_client = self.get_container_client(credential)
        full_path = f"{self.path}/{relative_path}" if relative_path else self.path
        if isinstance(container_client, FileSystemClient):
            with container_client.get_file_client(full_path) as file_client:
                return file_client.download_file().readall().decode()
        else:
            with container_client.get_blob_client(full_path) as blob_client:
                return blob_client.download_blob().readall().decode()

    def _read_local_file_content(self, relative_path: str = None) -> str:
        """Read file content from local path."""
        full_path = os.path.join(self._base_url, relative_path) if relative_path else self._base_url
        with open(full_path) as f:
            return f.read()

    def _is_local_folder_exists(self, relative_path: str = None) -> bool:
        full_path = os.path.join(self._base_url, relative_path) if relative_path else self._base_url
        return os.path.isdir(full_path)

    _SCHEME_MAP = {"blob&https": "wasbs", "blob&http": "wasb", "dfs&https": "abfss", "dfs&http": "abfs"}

    def _set_properties(self, ws: Workspace):
        url = urlparse(self._base_url)
        # TODO sovereign endpoint
        self._endpoint = "core.windows.net"
        if url.scheme in ["https", "http"]:
            pattern = r"(?P<scheme>http|https)://(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/" \
                      r"(?P<container>[^/]+)(?P<path>$|/(.*))"
            matches = re.match(pattern, self._base_url)
            if not matches:
                raise InvalidInputError(f"Unsupported uri as uri_folder: {self._base_url}")
            self.store_type = matches.group("store_type")
            self._scheme = StoreUrl._SCHEME_MAP[f"{self.store_type}&{matches.group('scheme')}"]
            self.account_name = matches.group("account_name")
            self.container_name = matches.group("container")
            self.path = matches.group("path").strip("/")
            self._datastore = None
        elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
            pattern = r"(?P<scheme>wasbs|abfss|wasb|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+)." \
                      r"(?P<store_type>blob|dfs).core.windows.net(?P<path>$|/(.*))"
            matches = re.match(pattern, self._base_url)
            if not matches:
                raise InvalidInputError(f"Unsupported uri as uri_folder: {self._base_url}")
            self._scheme = matches.group("scheme")
            self.store_type = matches.group("store_type")
            self.account_name = matches.group("account_name")
            self.container_name = matches.group("container")
            self.path = matches.group("path").strip("/")
            self._datastore = None
        elif url.scheme == "azureml":
            if ':' in url.path:  # azureml:<data_name>:<version> asset path
                # asset path should be translated to azureml or hdfs path in service, should not reach here
                raise InvalidInputError("AzureML asset path is not supported as uri_folder.")

            data_pattern0 = r"azureml://(subscriptions/([^/]+)/resource[gG]roups/([^/]+)/workspaces/([^/]+)/)?data/(?P<data>[^/]+)/versions/(?P<version>.+)"  # noqa: E501
            data_pattern1 = r"azureml://locations/([^/]+)/workspaces/([^/]+)/data/(?P<data>[^/]+)/versions/(?P<version>.+)"  # noqa: E501
            matches = re.match(data_pattern0, self._base_url) or re.match(data_pattern1, self._base_url)
            if matches:
                raise InvalidInputError(
                    "Seems you are using AzureML dataset v1 as input of Model Monitoring Job, but we only support "
                    "dataset v2. Please follow "
                    "https://learn.microsoft.com/en-us/azure/machine-learning/migrate-to-v2-assets-data?view=azureml-api-2"  # noqa: E501
                    "to upgrade your dataset to v2.")
            else:  # azureml datastore url, long or short form
                datastore_name, self.path = self._get_datastore_and_path_from_azureml_path()
                ws = ws or Run.get_context().experiment.workspace
                self._datastore = Datastore.get(ws, datastore_name)
                if self._datastore is None:
                    raise InvalidInputError(f"Datastore {datastore_name} not found in the workspace.")
                datastore_type = self._datastore.datastore_type
                if datastore_type not in ["AzureBlob", "AzureDataLakeGen2"]:
                    raise InvalidInputError("Only Azure Blob and Azure Data Lake Gen2 are supported, "
                                            f"but got {self._datastore.type}.")
                self.store_type = "dfs" if datastore_type == "AzureDataLakeGen2" else "blob"
                self._scheme = StoreUrl._SCHEME_MAP[f"{self.store_type}&{self._datastore.protocol}"]
                self.account_name = self._datastore.account_name
                self.container_name = self._datastore.container_name
        else:
            # file or other scheme, return original path directly
            self.account_name = None  # _account_name is None is the indicator that return the original base_path
            self._scheme = url.scheme
            self.path = url.path.strip("/")
            self._datastore = None  # indicator of no credential

    def _get_datastore_and_path_from_azureml_path(self) -> (str, str):
        """Get datastore name and path from azureml path."""
        pattern = r"azureml://(subscriptions/([^/]+)/resource[gG]roups/([^/]+)/workspaces/([^/]+)/)?datastores/(?P<datastore_name>[^/]+)/paths/(?P<path>.+)"  # noqa: E501
        matches = re.match(pattern, self._base_url)
        if not matches:
            raise InvalidInputError(f"Unsupported azureml uri: {self._base_url}")
        return matches.group("datastore_name"), matches.group("path").rstrip('/')

    def _is_secure(self):
        """Check if the store url is secure."""
        return self._scheme in ["wasbs", "abfss"]
