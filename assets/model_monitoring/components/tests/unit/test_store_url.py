# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for StoreUrl."""

import pytest
from unittest.mock import Mock
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import AzureSasCredential
from azure.storage.blob import ContainerClient, BlobServiceClient
from azure.storage.blob._shared.authentication import SharedKeyCredentialPolicy
from azure.storage.filedatalake import FileSystemClient
from model_data_collector_preprocessor.store_url import StoreUrl
from shared_utilities.momo_exceptions import InvalidInputError


@pytest.mark.unit
class TestStoreUrl:
    """Test class for StoreUrl."""

    @pytest.mark.parametrize(
        "store_base_url, expected_hdfs_path, expected_abfs_path",
        [
            (
                "https://my_account.blob.core.windows.net/my_container/path/to/folder",
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "https://my_account.dfs.core.windows.net/my_container/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "http://my_account.blob.core.windows.net/my_container/path/to/folder",
                "wasb://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder"
            ),
            (
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder"
            )
        ]
    )
    def test_store_url(self, store_base_url, expected_hdfs_path, expected_abfs_path):
        """Test StoreUrl constructor with http(s), abfs(s), wasb(s) and local file path."""
        store_url = StoreUrl(store_base_url)

        assert store_url.get_hdfs_url() == expected_hdfs_path
        assert store_url.get_abfs_url() == expected_abfs_path

        # before we support credential-less data, should raise InvalidInputError
        with pytest.raises(InvalidInputError):
            store_url.get_container_client()

    @pytest.mark.parametrize(
        "azureml_path, datastore_type, credential_type, expected_protocol, expected_hdfs_path, expected_abfs_path, "
        "expected_credential, expected_container_client",
        [
            (
                "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", "AccountKey", "https",
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "my_account_key",
                ContainerClient("https://my_account.blob.core.windows.net", "my_container",
                                SharedKeyCredentialPolicy("my_account", "my_account_key"))
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "AzureDataLakeGen2", "ServicePrincipal", "https",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                ClientSecretCredential("00000", "my_client_id", "my_client_secret"),
                FileSystemClient("https://my_account.dfs.core.windows.net", "my_container",
                                 ClientSecretCredential("00000", "my_client_id", "my_client_secret"))
            ),
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", "Sas", "https",
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                AzureSasCredential("my_sas_token"),
                ContainerClient("https://my_account.blob.core.windows.net", "my_container",
                                AzureSasCredential("my_sas_token"))
            ),
            (
                "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", "AccountKey", "http",
                "wasb://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "my_account_key",
                ContainerClient("http://my_account.blob.core.windows.net", "my_container",
                                SharedKeyCredentialPolicy("my_account", "my_account_key"))
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "AzureDataLakeGen2", None, "http",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                None,
                FileSystemClient("https://my_account.dfs.core.windows.net", "my_container")
            ),
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", None, "http",
                "wasb://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
                None,
                ContainerClient("http://my_account.blob.core.windows.net", "my_container")
            )
        ]
    )
    def test_store_url_with_azureml_path(
        self, azureml_path, datastore_type, credential_type, expected_protocol, expected_hdfs_path, expected_abfs_path,
        expected_credential, expected_container_client
    ):
        """Test StoreUrl constructor with azureml path."""
        mock_datastore = Mock(datastore_type=datastore_type, protocol=expected_protocol, endpoint="core.windows.net",
                              account_name="my_account", container_name="my_container")
        mock_datastore.name = "my_datastore"
        if datastore_type == "AzureBlob":
            mock_container_client = Mock(
                spec=ContainerClient, account_name="my_account", container_name="my_container",
                url=f"{expected_protocol}://my_account.blob.core.windows.net/my_container",
                credential=SharedKeyCredentialPolicy("my_account", "my_account_key")
                    if credential_type == "AccountKey" else expected_credential)
            mock_blob_svc = Mock(spec=BlobServiceClient)
            mock_blob_svc.get_container_client.side_effect = \
                lambda n: mock_container_client if n == "my_container" else None
            mock_datastore.credential_type = credential_type
            mock_datastore.blob_service = mock_blob_svc
            if credential_type is None:
                mock_datastore.account_key = None
                mock_datastore.sas_token = None
            elif credential_type == "AccountKey":
                mock_datastore.account_key = "my_account_key"
            elif credential_type == "Sas":
                mock_datastore.sas_token = "my_sas_token"
        elif datastore_type == "AzureDataLakeGen2":
            mock_datastore.client_id = "my_client_id" if credential_type else None
            mock_datastore.client_secret = "my_client_secret" if credential_type else None
            mock_datastore.tenant_id = "00000" if credential_type else None
        mock_ws = Mock(datastores={"my_datastore": mock_datastore})

        store_url = StoreUrl(azureml_path, mock_ws)

        assert store_url.get_hdfs_url() == expected_hdfs_path
        assert store_url.get_abfs_url() == expected_abfs_path
        if expected_credential:
            assert_credentials_are_equal(store_url.get_credential(), expected_credential)
            assert_container_clients_are_equal(store_url.get_container_client(), expected_container_client)
        else:
            # should raise InvalidInputError for credential less data
            with pytest.raises(InvalidInputError):
                store_url.get_credential()
            with pytest.raises(InvalidInputError):
                store_url.get_container_client()

    @pytest.mark.parametrize(
        "azureml_path",
        [
            "azureml://data/my_data/versions/1",
            "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/data/my_data/versions/1",
            "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/data/my_data/versions/2"
            "azureml://locations/my_loc/workspaces/my_ws/data/my_data/versions/1"
        ]
    )
    def test_store_url_with_azureml_data_url(self, azureml_path: str):
        """Test StoreUrl constructor with azureml data url."""
        with pytest.raises(InvalidInputError):
            _ = StoreUrl(azureml_path)

    @pytest.mark.parametrize(
        "base_url, relative_path, expected_root_path, expected_abfs_url",
        [
            ("https://my_account.blob.core.windows.net/my_container/path/to/base", "path/to/folder", "path/to/base",
             "abfss://my_container@my_account.dfs.core.windows.net/path/to/base/path/to/folder"),
            ("https://my_account.dfs.core.windows.net/my_container/path/to/base/", "folder", "path/to/base",
             "abfss://my_container@my_account.dfs.core.windows.net/path/to/base/folder"),
            ("wasbs://my_container@my_account.blob.core.windows.net/path/to/base", "/folder", "path/to/base",
             "abfss://my_container@my_account.dfs.core.windows.net/path/to/base/folder"),
            ("abfss://my_container@my_account.dfs.core.windows.net/path/to/base/", "folder/", "path/to/base",
             "abfss://my_container@my_account.dfs.core.windows.net/path/to/base/folder/"),
            ("http://my_account.blob.core.windows.net/my_container/path/to/base", "/folder/", "path/to/base",
             "abfs://my_container@my_account.dfs.core.windows.net/path/to/base/folder/"),
            ("http://my_account.dfs.core.windows.net/my_container/path/to/base/", "", "path/to/base",
             "abfs://my_container@my_account.dfs.core.windows.net/path/to/base"),
            ("wasb://my_container@my_account.blob.core.windows.net/path/to/base", "/", "path/to/base",
             "abfs://my_container@my_account.dfs.core.windows.net/path/to/base/"),
            ("https://my_account.blob.core.windows.net/my_container/base", "folder", "base",
             "abfss://my_container@my_account.dfs.core.windows.net/base/folder"),
            ("https://my_account.dfs.core.windows.net/my_container/base/", "/folder", "base",
             "abfss://my_container@my_account.dfs.core.windows.net/base/folder"),
            ("http://my_account.blob.core.windows.net/my_container/", "folder/", "",
             "abfs://my_container@my_account.dfs.core.windows.net/folder/"),
            ("http://my_account.dfs.core.windows.net/my_container", "", "",
             "abfs://my_container@my_account.dfs.core.windows.net"),
            ("abfs://my_container@my_account.dfs.core.windows.net", "/", "",
             "abfs://my_container@my_account.dfs.core.windows.net/"),
            ("file:///path/to/base", "path/to/folder", "path/to/base", "file:///path/to/base/path/to/folder"),
            (r"d:\path\to\base", "folder/to/path", r"\path\to\base", r"d:\path\to\base/folder/to/path")
        ]
    )
    def test_get_abfs_url_with_relative_path(self, base_url, relative_path, expected_root_path, expected_abfs_url):
        """Test get_abfs_url() with relative path."""
        store_url = StoreUrl(base_url)

        assert store_url.path == expected_root_path
        assert store_url.get_abfs_url(relative_path) == expected_abfs_url


def assert_credentials_are_equal(credential1, credential2):
    """Assert 2 credentials are equal."""
    if credential1 is None:
        assert credential2 is None
    elif isinstance(credential1, ClientSecretCredential):
        assert isinstance(credential2, ClientSecretCredential)
        assert credential1._tenant_id == credential2._tenant_id
        assert credential1._client_id == credential2._client_id
        assert credential1._client_credential == credential2._client_credential
    elif isinstance(credential1, AzureSasCredential):
        assert isinstance(credential2, AzureSasCredential)
        assert credential1.signature == credential2.signature
    elif isinstance(credential1, str):  # account key
        assert credential1 == credential2
    elif isinstance(credential1, SharedKeyCredentialPolicy):
        assert isinstance(credential2, SharedKeyCredentialPolicy)
        assert credential1.account_name == credential2.account_name
        assert credential1.account_key == credential2.account_key
    elif isinstance(credential1, DefaultAzureCredential):
        assert isinstance(credential2, DefaultAzureCredential)
    else:
        raise NotImplementedError(f"Unsupported credential type: {type(credential1)}")


def assert_container_clients_are_equal(container_client1, container_client2):
    """Assert 2 container clients are equal."""
    def assert_container_clients_properties_are_equal():
        assert container_client1.account_name == container_client2.account_name
        assert container_client1.url == container_client2.url
        assert_credentials_are_equal(container_client1.credential, container_client2.credential)

    if container_client1 is None:
        assert container_client2 is None
    elif isinstance(container_client1, ContainerClient):
        assert isinstance(container_client2, ContainerClient)
        assert container_client1.container_name == container_client2.container_name
        assert_container_clients_properties_are_equal()
    elif isinstance(container_client1, FileSystemClient):
        assert isinstance(container_client2, FileSystemClient)
        assert container_client1.file_system_name == container_client2.file_system_name
        assert_container_clients_properties_are_equal
    else:
        raise NotImplementedError(f"Unsupported container client type: {type(container_client1)}")
