# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from urllib.parse import urlparse
import re
from typing import Union
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import AzureSasCredential
from azure.storage.blob import ContainerClient
from azure.storage.filedatalake import FileSystemClient
from azureml.core import Workspace, Run
from shared_utilities.momo_exceptions import InvalidInputError


class StoreUrl:
    """Helper class to Convert base_path to HDFS path."""

    def __init__(self, base_url: str, ws: Workspace = None):
        self._base_url = base_url
        self._set_properties(ws)

    def get_hdfs_url(self, relative_path: str = None) -> str:
        """
        Get HDFS url for this store url.

        :param relative_path: relative path to the base path
        :return: HDFS url, will be abfs(s) path for gen2 and wasb(s) for blob store
        """
        if not self.account_name:
            return self._base_url

        hdfs_url = f"{self._scheme}://{self.container_name}@{self.account_name}.{self._store_type}." \
                   f"{self._endpoint}/{self._path}"
        if relative_path:
            hdfs_url = f"{hdfs_url}/{relative_path.lstrip('/')}"
        return hdfs_url

    def get_abfs_url(self, relative_path: str = None) -> str:
        """
        Get abfs url for the store url.

        :param relative_path: relative path to the base path
        :return: always abfs(s) url, to access append blob in blob store
        """
        if not self.account_name:
            return self._base_url

        scheme = "abfss" if self._is_secure() else "abfs"
        store_type = "dfs"
        abfs_url = f"{scheme}://{self.container_name}@{self.account_name}.{store_type}.{self._endpoint}/{self._path}"
        if relative_path:
            abfs_url = f"{abfs_url}/{relative_path.lstrip('/')}"
        return abfs_url

    def get_credential(self) -> Union[str, ClientSecretCredential, AzureSasCredential, None]:
        """Get credential for this store url."""
        def get_default_credential():
            return DefaultAzureCredential() if self._is_secure() else None

        if not self._datastore:
            return get_default_credential()
        elif self._datastore.datastore_type == "AzureBlob":
            if self._datastore.credential_type == "AccountKey":
                return self._datastore.account_key
            elif self._datastore.credential_type == "Sas":
                return AzureSasCredential(self._datastore.sas_token)
            elif self._datastore.credential_type is None:
                return get_default_credential()  # credential less datastore
            else:
                raise InvalidInputError(f"Unsupported credential type: {self._datastore.credential_type}, "
                                        "only AccountKey and Sas are supported.")
        elif self._datastore.datastore_type == "AzureDataLakeGen2":
            if self._datastore.tenant_id and self._datastore.client_id:
                return ClientSecretCredential(tenant_id=self._datastore.tenant_id, client_id=self._datastore.client_id,
                                              client_secret=self._datastore.client_secret)
            else:
                return get_default_credential()  # credential less datastore
        else:
            raise InvalidInputError(f"Unsupported datastore type: {self._datastore.datastore_type}, "
                                    "only Azure Blob and Azure Data Lake Gen2 are supported.")

    def get_container_client(self) -> Union[FileSystemClient, ContainerClient, None]:
        """Get container client for this store url."""
        if not self.account_name:
            # local or not supported store type
            return None

        # blob, has cred datastore
        if self._store_type == "blob" and self._datastore and self._datastore.credential_type:
            return self._datastore.blob_service.get_container_client(self.container_name)
        # TODO fallback to DefaultAzureCredential for credential less datastore for now, may need better fallback logic
        credential = self.get_credential()
        account_url_scheme = "https" if self._is_secure() else "http"
        if self._store_type == "blob":
            return ContainerClient(account_url=f"{account_url_scheme}://{self.account_name}.blob.core.windows.net",
                                   container_name=self.container_name, credential=credential)
        elif self._store_type == "dfs":
            return FileSystemClient(account_url=f"{account_url_scheme}://{self.account_name}.dfs.core.windows.net",
                                    file_system_name=self.container_name, credential=credential)
        else:
            raise InvalidInputError(f"Unsupported store type: {self._store_type}, only blob and dfs are supported.")

    _SCHEME_MAP = {"blob&https": "wasbs", "blob&http": "wasb", "dfs&https": "abfss", "dfs&http": "abfs"}

    def _set_properties(self, ws: Workspace):
        url = urlparse(self._base_url)
        # TODO sovereign endpoint
        self._endpoint = "core.windows.net"
        if url.scheme in ["https", "http"]:
            pattern = r"(?P<scheme>http|https)://(?P<account_name>[^\.]+).(?P<store_type>blob|dfs).core.windows.net/" \
                      r"(?P<container>[^/]+)/(?P<path>.+)"
            matches = re.match(pattern, self._base_url)
            if not matches:
                raise InvalidInputError(f"Unsupported uri as uri_folder: {self._base_url}")
            self._store_type = matches.group("store_type")
            # use abfss to access both blob and dfs, to workaround the append block issue
            self._scheme = StoreUrl._SCHEME_MAP[f"{self._store_type}&{matches.group('scheme')}"]
            self.account_name = matches.group("account_name")
            self.container_name = matches.group("container")
            self._path = matches.group("path").strip("/")
            self._datastore = None
            # filesystem_client = _get_filesystem_client(account_name, container_name, matches.group("store_type"),
            #                                            spark, credential)
            # return f"{scheme}://{container_name}@{account_name}.{store_type}.core.windows.net/{path}", None
        elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
            pattern = r"(?P<scheme>wasbs|abfss|wasb|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+)." \
                      r"(?P<store_type>blob|dfs).core.windows.net/(?P<path>.+)"
            matches = re.match(pattern, self._base_url)
            if not matches:
                raise InvalidInputError(f"Unsupported uri as uri_folder: {self._base_url}")
            self._scheme = matches.group("scheme")
            self._store_type = matches.group("store_type")
            self.account_name = matches.group("account_name")
            self.container_name = matches.group("container")
            self._path = matches.group("path").strip("/")
            self._datastore = None
            # filesystem_client = _get_filesystem_client(account_name, container_name, matches.group("store_type"),
            #                                            spark, credential)
            # return f"{scheme}://{container_name}@{account_name}.{store_type}.core.windows.net/{path}", None
        elif url.scheme == "azureml":
            if ':' in url.path:  # azureml asset path
                # asset path should be translated to azureml or hdfs path in service, should not reach here
                raise InvalidInputError("AzureML asset path is not supported as uri_folder.")
            else:  # azureml long or short form
                datastore_name, self._path = self._get_datastore_and_path_from_azureml_path()
                ws = ws or Run.get_context().experiment.workspace
                self._datastore = ws.datastores.get(datastore_name)
                datastore_type = self._datastore.datastore_type
                if datastore_type not in ["AzureBlob", "AzureDataLakeGen2"]:
                    raise InvalidInputError("Only Azure Blob and Azure Data Lake Gen2 are supported, "
                                            f"but got {self._datastore.type}.")
                self._store_type = "dfs" if datastore_type == "AzureDataLakeGen2" else "blob"
                self._scheme = StoreUrl._SCHEME_MAP[f"{self._store_type}&{self._datastore.protocol}"]
                self.account_name = self._datastore.account_name
                self.container_name = self._datastore.container_name
                # container_client = _get_container_client(datastore)
                # return f"{scheme}://{container_name}@{account_name}.{store_type}.core.windows.net/{path}", container_client
        else:
            # file or other scheme, return original path directly
            self.account_name = None  # _account_name is None is the indicator that return the original base_path
            self._datastore = None  # indicator of no credential

    def _get_datastore_and_path_from_azureml_path(self) -> (str, str):
        """Get datastore name and path from azureml path."""
        start_idx = self._base_url.find('/datastores/')
        end_idx = self._base_url.find('/paths/')
        return self._base_url[start_idx+12:end_idx], self._base_url[end_idx+7:].rstrip('/')

    def _is_secure(self):
        """Check if the store url is secure."""
        return self._scheme in ["wasbs", "abfss"]
