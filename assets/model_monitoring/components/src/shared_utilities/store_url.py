# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Contains StoreUrl class."""

from urllib.parse import urlparse
import os
import re
import glob
from typing import Union, Tuple
import fnmatch
from azure.identity import ClientSecretCredential
from azure.core.credentials import AzureSasCredential
from azure.core.exceptions import HttpResponseError
from azure.storage.blob import ContainerClient
from azure.storage.filedatalake import FileSystemClient
from azureml.core import Workspace, Run, Datastore
from azureml.exceptions import UserErrorException
from shared_utilities.constants import MISSING_OBO_CREDENTIAL_HELPFUL_ERROR_MESSAGE, UAI_MISS_PERMISSION_ERROR_MESSAGE
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
            raise InvalidInputError(f"{self._base_url} is not an azureml url.")
        url = (f"azureml://subscriptions/{self._datastore.workspace.subscription_id}/resourceGroups"
               f"/{self._datastore.workspace.resource_group}/workspaces/{self._datastore.workspace.name}"
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

    def get_credential(self, validate_aml_obo_credential: bool = True) -> Union[
            str, ClientSecretCredential, AzureSasCredential, None]:
        """Get credential for this store url."""
        def valid_aml_obo_credential():
            """Validate AzureMLOnBehalfOfCredential can be used in the environment before returns it."""
            if not self._is_secure():
                raise InvalidInputError(
                    "Unsecure credential-less data is not supported. "
                    "Please use either a secure or a credential url for the StoreUrl.")
            try:
                from azure.ai.ml.identity import AzureMLOnBehalfOfCredential, CredentialUnavailableError
                aml_obo_credential = AzureMLOnBehalfOfCredential()
                if validate_aml_obo_credential:
                    try:
                        _ = aml_obo_credential.get_token("https://management.azure.com/.default")
                        return aml_obo_credential
                    except CredentialUnavailableError as cue:
                        raise InvalidInputError(MISSING_OBO_CREDENTIAL_HELPFUL_ERROR_MESSAGE.format(message=str(cue)))
                else:
                    return aml_obo_credential
            except ModuleNotFoundError:
                print("Failed to import AzureMLOnBehalfOfCredential. "
                      "Cannot check if unsecure URL was used with token credential. "
                      "Continuing and expecting no failures...")

        if not self._datastore:
            print("Using AML OBO credential from StoreUrl.get_credential() because the internal datastore is None.")
            return valid_aml_obo_credential()
        elif self._datastore.datastore_type == "AzureBlob":
            if self._datastore.credential_type == "AccountKey":
                print("Using acount key credential from StoreUrl.get_credential() with blob datastore.")
                return self._datastore.account_key
            elif self._datastore.credential_type == "Sas":
                print("Using SAS token credential from StoreUrl.get_credential() with blob datastore.")
                return AzureSasCredential(self._datastore.sas_token)
            elif self._datastore.credential_type is None or self._datastore.credential_type == "None":
                print("Using AML OBO credential from StoreUrl.get_credential() with blob datastore "
                      "whose credential type is null.")
                return valid_aml_obo_credential()
            else:
                raise InvalidInputError(f"Unsupported credential type: {self._datastore.credential_type}, "
                                        "only AccountKey and Sas are supported.")
        elif self._datastore.datastore_type == "AzureDataLakeGen2":
            if self._datastore.tenant_id and self._datastore.client_id and self._datastore.client_secret:
                print("Using Client Secret credential from StoreUrl.get_credential() with Gen2 datastore.")
                return ClientSecretCredential(tenant_id=self._datastore.tenant_id, client_id=self._datastore.client_id,
                                              client_secret=self._datastore.client_secret)
            else:
                print("Using AML OBO credential from StoreUrl.get_credential() with Gen2 datastore "
                      "whose saved credential info does not have client secret available to authenticate with.")
                return valid_aml_obo_credential()
        else:
            raise InvalidInputError(f"Unsupported datastore type: {self._datastore.datastore_type}, "
                                    "only Azure Blob and Azure Data Lake Gen2 are supported.")

    def is_credentials_less(self) -> bool:
        """Check if the store url is credential less."""
        # TODO: remove after we figure out cache failure issues.
        # Should be able to import AzureMLOnBehalfOfCredential to check the class w/o issues.
        credential = None
        try:
            credential = self.get_credential()
            from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
            return credential is None or isinstance(credential, AzureMLOnBehalfOfCredential)
        except ModuleNotFoundError:
            print(
                "Failed to import AzureMLOnBehalfOfCredential to check credential class instance. "
                "Defaulting to None credential-check solely.")
            return credential is None

    def get_container_client(
            self,
            credential: Union[str, AzureSasCredential, ClientSecretCredential, None] = None,
            validate_aml_obo_credential: bool = True
            ) -> Union[FileSystemClient, ContainerClient, None]:
        """
        Get container client for this store url.

        :param credential: if provided, it contains the credential to authorize the container to access the data;
        if not provided, will retrieve credential from datastore,
        if datastore absent or is credential-less, use Azureml OBO credential.
        It's a special handling for access dataref file in executors.
        """
        if not self.account_name:
            # local or not supported store type
            return None

        # blob, has cred datastore
        if self.store_type == "blob" and self._datastore \
                and (self._datastore.credential_type and self._datastore.credential_type != "None"):
            return self._datastore.blob_service.get_container_client(self.container_name)

        # fallback to AzureMLOnBehalfOfCredential for credential less datastore for now.
        # Requires that we submit MoMo component with managed identity or will fail later on.
        credential = credential or self.get_credential(validate_aml_obo_credential)
        account_url_scheme = "https" if self._is_secure() else "http"

        if self.store_type == "blob":
            return ContainerClient(account_url=f"{account_url_scheme}://{self.account_name}.blob.{self._endpoint}",
                                   container_name=self.container_name, credential=credential)
        elif self.store_type == "dfs":
            return FileSystemClient(account_url=f"{account_url_scheme}://{self.account_name}.dfs.{self._endpoint}",
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

    def any_files(self, relative_path_pattern: str,
                  container_client: Union[FileSystemClient, ContainerClient, None] = None) -> bool:
        """Check if file matching the `relative_path_pattern` exists, wildcard supported."""
        def file_name_match(file_name: str, pattern: str) -> bool:
            # fnmatch will match /a/b.txt with /*.txt, but we want it to return False
            # so split the path and pattern and then match with fnmatch within each section
            path_sections = file_name.split("/")
            pattern_sections = pattern.split("/")
            if len(path_sections) != len(pattern_sections):
                return False
            return all(fnmatch.fnmatch(file_name, pattern)
                       for file_name, pattern in zip(path_sections, pattern_sections))

        def any_files(file_names: list, pattern):
            return any(file_name_match(file_name, pattern) for file_name in file_names)

        base_path = self._base_url.replace('\\', '/') if self.is_local_path() else self.path
        full_path_pattern = f"{base_path.rstrip('/')}/{relative_path_pattern.strip('/')}"
        # find the non-wildcard part of the path
        pattern_sections = full_path_pattern.split("/")
        for idx in range(len(pattern_sections)):
            if "*" in pattern_sections[idx] or '?' in pattern_sections[idx]:
                break
        non_wildcard_path = ("/".join(pattern_sections[:idx]) + "/").lstrip("/")  # lstrip to handle idx == 0
        path_pattern = "/".join(pattern_sections[idx:])
        # match the wildcard part of the path
        container_client = container_client or self.get_container_client()
        if not container_client:  # local
            return any(glob.iglob(full_path_pattern))
        try:
            if isinstance(container_client, FileSystemClient):  # gen2
                if container_client.get_directory_client(non_wildcard_path).exists():
                    paths = container_client.get_paths(non_wildcard_path, True)
                    file_names = [path.name[len(non_wildcard_path):] for path in paths if not path.is_directory]
                else:
                    return False
            else:  # blob
                blobs = container_client.list_blobs(name_starts_with=non_wildcard_path)
                file_names = [blob.name[len(non_wildcard_path):] for blob in blobs]
        except HttpResponseError as hre:
            if hre.status_code == 403:
                raise InvalidInputError(UAI_MISS_PERMISSION_ERROR_MESSAGE)
            else:
                raise hre
        return any_files(file_names, path_pattern)

    def is_local_path(self) -> bool:
        """Check if the store url is a local path."""
        if not self._base_url:
            return False
        if os.path.isdir(self._base_url) or os.path.isfile(self._base_url) \
                or re.match(r"^[a-zA-Z]:[/\\]", self._base_url):
            return True
        url = urlparse(self._base_url)
        return url.scheme is None or url.scheme == "file" or url.scheme == ""

    def read_file_content(self, relative_path: str = None,
                          credential: Union[str, AzureSasCredential, ClientSecretCredential, None] = None) -> str:
        """Read file content from the store."""
        if self.is_local_path():
            return self._read_local_file_content(relative_path)

        container_client = self.get_container_client(credential)
        full_path = f"{self.path}/{relative_path.strip('/')}" if relative_path else self.path
        if isinstance(container_client, FileSystemClient):
            with container_client.get_file_client(full_path) as file_client:
                return file_client.download_file().readall().decode()
        else:
            with container_client.get_blob_client(full_path) as blob_client:
                return blob_client.download_blob().readall().decode()

    def write_file(self, file_content: Union[str, bytes], relative_path: str = None, overwrite: bool = False,
                   credential: Union[str, AzureSasCredential, ClientSecretCredential, None] = None) -> dict:
        """Upload file to the store."""
        if self.is_local_path():
            return {"bytes_written": self._write_local_file(file_content, relative_path)}

        container_client = self.get_container_client(credential)
        full_path = f"{self.path}/{relative_path.strip('/')}" if relative_path else self.path
        if isinstance(container_client, FileSystemClient):
            with container_client.get_file_client(full_path) as file_client:
                if not file_client.exists():
                    # when writing to a Gen2 storage location,
                    # if the file doesn't exist we can't just use upload_data() as it will throw errors.
                    # Instead use this work around to create the file, append data, and flush the data commit.
                    return self._create_file_and_append_content(container_client, file_content, full_path)
                # upload_data() works fine with Gen2 if the file exists.
                return file_client.upload_data(file_content, overwrite=overwrite)
        else:
            with container_client.get_blob_client(full_path) as blob_client:
                return blob_client.upload_blob(file_content, overwrite=overwrite)

    @staticmethod
    def _normalize_local_path(local_path: str) -> str:
        """Normalize local path."""
        return local_path[7:] if local_path.startswith("file://") else local_path

    def _read_local_file_content(self, relative_path: str = None) -> str:
        """Read file content from local path."""
        base_url = StoreUrl._normalize_local_path(self._base_url)
        full_path = os.path.join(base_url, relative_path) if relative_path else base_url
        with open(full_path) as f:
            return f.read()

    def _create_file_and_append_content(
            self, container_client: FileSystemClient,
            file_content: Union[str, bytes], full_path: str):
        """Create a new file in FilSystemClient and append the file_content to the new file."""
        print("Requested file does not exist. Create file and append data...")
        with container_client.create_file(full_path) as file_client:
            file_client.append_data(file_content, offset=0)

            content_length = len(file_content)
            return file_client.flush_data(content_length)

    def _write_local_file(self, file_content: Union[str, bytes], relative_path: str = None) -> int:
        """Write file to local path."""
        base_url = StoreUrl._normalize_local_path(self._base_url)
        full_path = os.path.join(base_url, relative_path) if relative_path else base_url
        # create folder if it does not exist
        dir_path = os.path.dirname(full_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        with open(full_path, "w" if isinstance(file_content, str) else "wb") as f:
            return f.write(file_content)

    def _is_local_folder_exists(self, relative_path: str = None) -> bool:
        full_path = os.path.join(self._base_url, relative_path) if relative_path else self._base_url
        return os.path.isdir(full_path)

    _SCHEME_MAP = {"blob&https": "wasbs", "blob&http": "wasb", "dfs&https": "abfss", "dfs&http": "abfs"}
    _STORAGE_ENDPOINTS_REGEX = r"core\.windows\.net|core\.usgovcloudapi\.net|core\.chinacloudapi\.cn|core\.microsoft\.scloud|core\.eaglex\.ic\.gov"  # noqa: E501

    def _set_properties(self, ws: Workspace):
        url = urlparse(self._base_url)
        if url.scheme in ["https", "http"]:
            pattern = r"(?P<scheme>http|https)://(?P<account_name>[^\.]+)\.(?P<store_type>blob|dfs)\." \
                      f"(?P<endpoint>{StoreUrl._STORAGE_ENDPOINTS_REGEX})/" \
                      r"(?P<container>[^/]+)(?P<path>$|/(.*))"
            matches = re.match(pattern, self._base_url)
            if not matches:
                raise InvalidInputError(f"Unsupported uri as uri_folder: {self._base_url}")
            self.store_type = matches.group("store_type")
            self._scheme = StoreUrl._SCHEME_MAP[f"{self.store_type}&{matches.group('scheme')}"]
            self.account_name = matches.group("account_name")
            self.container_name = matches.group("container")
            self.path = matches.group("path").strip("/")
            self._endpoint = matches.group("endpoint")
            self._datastore = None
        elif url.scheme in ["wasbs", "wasb", "abfss", "abfs"]:
            pattern = r"(?P<scheme>wasbs|abfss|wasb|abfs)://(?P<container>[^@]+)@(?P<account_name>[^\.]+)\." \
                      fr"(?P<store_type>blob|dfs)\.(?P<endpoint>{StoreUrl._STORAGE_ENDPOINTS_REGEX})" \
                      r"(?P<path>$|/(.*))"
            matches = re.match(pattern, self._base_url)
            if not matches:
                raise InvalidInputError(f"Unsupported uri as uri_folder: {self._base_url}")
            self._scheme = matches.group("scheme")
            self.store_type = matches.group("store_type")
            self.account_name = matches.group("account_name")
            self.container_name = matches.group("container")
            self.path = matches.group("path").strip("/")
            self._endpoint = matches.group("endpoint")
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
                try:
                    self._datastore = Datastore.get(ws, datastore_name)
                except UserErrorException:
                    raise InvalidInputError(f"Datastore {datastore_name} not found in the workspace.")
                datastore_type = self._datastore.datastore_type
                if datastore_type not in ["AzureBlob", "AzureDataLakeGen2"]:
                    raise InvalidInputError("Only Azure Blob and Azure Data Lake Gen2 are supported, "
                                            f"but got {datastore_type}.")
                self.store_type = "dfs" if datastore_type == "AzureDataLakeGen2" else "blob"
                self._scheme = StoreUrl._SCHEME_MAP[f"{self.store_type}&{self._datastore.protocol}"]
                self.account_name = self._datastore.account_name
                self.container_name = self._datastore.container_name
                self._endpoint = self._datastore.endpoint
        else:
            # file or other scheme, return original path directly
            self.account_name = None  # _account_name is None is the indicator that return the original base_path
            self._scheme = url.scheme
            self.path = url.path.strip("/")
            self._datastore = None  # indicator of no credential

    def _get_datastore_and_path_from_azureml_path(self) -> Tuple[str, str]:
        """Get datastore name and path from azureml path."""
        pattern = r"azureml://(subscriptions/([^/]+)/resource[gG]roups/([^/]+)/workspaces/([^/]+)/)?datastores/(?P<datastore_name>[^/]+)/paths/(?P<path>.+)"  # noqa: E501
        matches = re.match(pattern, self._base_url)
        if not matches:
            raise InvalidInputError(f"Unsupported azureml uri: {self._base_url}")
        return matches.group("datastore_name"), matches.group("path").rstrip('/')

    def _is_secure(self):
        """Check if the store url is secure."""
        return self._scheme in ["wasbs", "abfss"]
