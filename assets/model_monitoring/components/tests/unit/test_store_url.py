# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for StoreUrl."""

import pytest
import os
from unittest.mock import Mock, patch
from azure.ai.ml.identity import AzureMLOnBehalfOfCredential
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import AzureSasCredential
from azure.storage.blob import ContainerClient, BlobServiceClient
from azure.storage.blob._shared.authentication import SharedKeyCredentialPolicy
from azure.storage.filedatalake import FileSystemClient, DataLakeDirectoryClient
from azureml.core import Datastore
from azureml.exceptions import UserErrorException
from shared_utilities.store_url import StoreUrl
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
            ),
            (
                "abfss://my_container@my_account.dfs.core.usgovcloudapi.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.usgovcloudapi.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.usgovcloudapi.net/path/to/folder"
            ),
            (
                "https://my_account.blob.core.chinacloudapi.cn/my_container/path/to/folder",
                "wasbs://my_container@my_account.blob.core.chinacloudapi.cn/path/to/folder",
                "abfss://my_container@my_account.dfs.core.chinacloudapi.cn/path/to/folder"
            ),
        ]
    )
    def test_store_url(self, store_base_url, expected_hdfs_path, expected_abfs_path):
        """Test StoreUrl constructor with http(s), abfs(s), wasb(s) and local file path."""
        store_url = StoreUrl(store_base_url)

        assert store_url.get_hdfs_url() == expected_hdfs_path
        assert store_url.get_abfs_url() == expected_abfs_path

        # We support credential-less with secure store url.
        if store_url._is_secure():
            container = store_url.get_container_client(validate_aml_obo_credential=False)
            assert container is not None
            assert isinstance(container.credential, AzureMLOnBehalfOfCredential)
        else:
            with pytest.raises(InvalidInputError,
                               match=r"Unsecure credential-less data is not supported...*"):
                store_url.get_container_client()

    PUBLIC_ENDPOINT = "core.windows.net"
    USGOV_ENDPOINT = "core.usgovcloudapi.net"
    CHINA_ENDPOINT = "core.chinacloudapi.cn"

    @pytest.mark.parametrize(
        "azureml_path, datastore_type, endpoint, credential_type, relative_path, expected_protocol, "
        "expected_hdfs_path, expected_abfs_path, expected_relative_path, expected_credential, "
        "expected_container_client",
        [
            (
                "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", PUBLIC_ENDPOINT, "AccountKey", None, "https",
                f"wasbs://my_container@my_account.blob.{PUBLIC_ENDPOINT}/path/to/folder",
                f"abfss://my_container@my_account.dfs.{PUBLIC_ENDPOINT}/path/to/folder",
                "", "my_account_key",
                ContainerClient(f"https://my_account.blob.{PUBLIC_ENDPOINT}", "my_container",
                                SharedKeyCredentialPolicy("my_account", "my_account_key"))
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "AzureDataLakeGen2", PUBLIC_ENDPOINT, "ServicePrincipal", "rpath", "https",
                f"abfss://my_container@my_account.dfs.{PUBLIC_ENDPOINT}/path/to/folder",
                f"abfss://my_container@my_account.dfs.{PUBLIC_ENDPOINT}/path/to/folder",
                "/rpath",
                ClientSecretCredential("00000", "my_client_id", "my_client_secret"),
                FileSystemClient(f"https://my_account.dfs.{PUBLIC_ENDPOINT}", "my_container",
                                 ClientSecretCredential("00000", "my_client_id", "my_client_secret"))
            ),
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", USGOV_ENDPOINT, "Sas", "rpath/to/target", "https",
                f"wasbs://my_container@my_account.blob.{USGOV_ENDPOINT}/path/to/folder",
                f"abfss://my_container@my_account.dfs.{USGOV_ENDPOINT}/path/to/folder",
                "/rpath/to/target",
                AzureSasCredential("my_sas_token"),
                ContainerClient(f"https://my_account.blob.{USGOV_ENDPOINT}", "my_container",
                                AzureSasCredential("my_sas_token"))
            ),
            (
                "azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", CHINA_ENDPOINT, "AccountKey", "", "http",
                f"wasb://my_container@my_account.blob.{CHINA_ENDPOINT}/path/to/folder",
                f"abfs://my_container@my_account.dfs.{CHINA_ENDPOINT}/path/to/folder",
                "", "my_account_key",
                ContainerClient(f"http://my_account.blob.{CHINA_ENDPOINT}", "my_container",
                                SharedKeyCredentialPolicy("my_account", "my_account_key"))
            ),
            (
                "azureml://datastores/my_datastore/paths/path/to/folder",
                "AzureDataLakeGen2", PUBLIC_ENDPOINT, None, "/", "https",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "/", AzureMLOnBehalfOfCredential(),
                FileSystemClient("https://my_account.dfs.core.windows.net", "my_container",
                                 AzureMLOnBehalfOfCredential())
            ),
            (
                "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
                "/paths/path/to/folder",
                "AzureBlob", PUBLIC_ENDPOINT, "None", "/rpath", "https",
                "wasbs://my_container@my_account.blob.core.windows.net/path/to/folder",
                "abfss://my_container@my_account.dfs.core.windows.net/path/to/folder",
                "/rpath", AzureMLOnBehalfOfCredential(),
                ContainerClient("https://my_account.blob.core.windows.net", "my_container",
                                AzureMLOnBehalfOfCredential())
            ),
            # TODO: Update this UT with how the unsecure URL should work with credential-less. Or can we remove it?
            # (
            #     "azureml://datastores/my_datastore/paths/path/to/folder",
            #     "AzureDataLakeGen2", None, "/", "http",
            #     "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
            #     "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
            #     "/", None,
            #     FileSystemClient("https://my_account.dfs.core.windows.net", "my_container")
            # ),
            # (
            #     "azureml://subscriptions/sub_id/resourcegroups/my_rg/workspaces/my_ws/datastores/my_datastore"
            #     "/paths/path/to/folder",
            #     "AzureBlob", "None", "/rpath", "http",
            #     "wasb://my_container@my_account.blob.core.windows.net/path/to/folder",
            #     "abfs://my_container@my_account.dfs.core.windows.net/path/to/folder",
            #     "/rpath", None,
            #     ContainerClient("http://my_account.blob.core.windows.net", "my_container")
            # )
        ]
    )
    def test_store_url_with_azureml_path(
        self, azureml_path, datastore_type, endpoint, credential_type, relative_path, expected_protocol,
        expected_hdfs_path, expected_abfs_path, expected_relative_path, expected_credential, expected_container_client
    ):
        """Test StoreUrl constructor with azureml path."""
        mock_ws = Mock(subscription_id="sub_id", resource_group="my_rg")
        mock_ws.name = "my_ws"
        mock_datastore = Mock(datastore_type=datastore_type, protocol=expected_protocol, endpoint=endpoint,
                              account_name="my_account", container_name="my_container", subscription_id="store_sub_id",
                              resource_group="store_rg", workspace=mock_ws)
        mock_datastore.name = "my_datastore"
        if datastore_type == "AzureBlob":
            mock_container_client = Mock(
                spec=ContainerClient, account_name="my_account", container_name="my_container",
                url=f"{expected_protocol}://my_account.blob.{endpoint}/my_container",
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

        with patch.object(Datastore, "get", return_value=mock_datastore):
            store_url = StoreUrl(azureml_path, mock_ws)

        assert store_url.get_hdfs_url() == expected_hdfs_path
        assert store_url.get_abfs_url() == expected_abfs_path
        assert store_url.get_azureml_url(relative_path) == \
            ("azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore"
             f"/paths/path/to/folder{expected_relative_path}")
        assert store_url._endpoint == endpoint

        if expected_credential:
            assert_credentials_are_equal(store_url.get_credential(validate_aml_obo_credential=False),
                                         expected_credential)
            assert_container_clients_are_equal(store_url.get_container_client(validate_aml_obo_credential=False),
                                               expected_container_client)
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

    def test_store_url_datastore_not_found(self):
        """Test StoreUrl constructor, in case datastore not found ."""
        mock_ws = Mock()

        with patch.object(Datastore, "get", side_effect=UserErrorException("Datastore not found.")):
            with pytest.raises(InvalidInputError, match=r"Datastore my_datastore not found .*"):
                _ = StoreUrl("azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws"
                             "/datastores/my_datastore/paths/path/to/folder",
                             mock_ws)

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

    @pytest.mark.parametrize(
        "path, is_local_path",
        [
            ("./path/to/folder", True),
            ("../path/to/folder", True),
            ("path/to/file.parquet", True),
            ("/path/to/folder", True),
            ("./my_data.csv", True),
            ("my_data.csv", True),
            ("C:/path/to/folder", True),
            (r"D:\path\to\file.csv", True),
            ("file:///path/to/folder", True),
            ("file://path/to/data.parquet", True),
            ("https://my_account.blob.core.windows.net/my_container/path/to/folder", False),
            ("abfss://my_container@my_account.dfs.core.windows.net/path/to/folder", False),
            ("azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_ws/datastores/my_datastore/paths/path/to/folder", False),  # noqa: E501
            ("azureml://datastores/my_datastore/paths/path/to/data.parquet", False),
        ]
    )
    def test_is_local_path(self, path, is_local_path):
        """Test is_local_path()."""
        mock_datastore = Mock(datastore_type="AzureBlob", protocol="https", endpoint="core.windows.net",
                              account_name="my_account", container_name="my_container")
        mock_datastore.name = "my_datastore"
        mock_datastore.credential_type = "Sas"
        mock_datastore.sas_token = "my_sas_token"
        mock_ws = Mock()

        with patch.object(Datastore, "get", return_value=mock_datastore):
            store_url = StoreUrl(path, mock_ws)
            assert store_url.is_local_path() == is_local_path

    PATHS = [
        "my/folder/folder1/folder2/file1.jsonl",
        "my/folder/folder1/file2.jsonl",
        "my/folder/folder3/file3.csv",
        "my/folder/file4.jsonl",
        "my/folder/folder1/folder2",
        "my/folder/folder1",
        "my/folder/folder3",
        "my/folder/folder4",  # empty folder
        "my/folder/folder4/folder5"
    ]

    @pytest.mark.parametrize(
        "path_names, base_folder, pattern, expected_folder, expected_result",
        [
            (PATHS, "/my/folder", "/*.jsonl", "my/folder/", True),
            (PATHS, "/my/folder", "*/*.jsonl", "my/folder/", True),
            (PATHS, "/my/folder", "*/*/*.jsonl", "my/folder/", True),
            (PATHS, "/my/folder", "*/*/*/*.jsonl", "my/folder/", False),
            (PATHS, "/my/folder", "/folder1/*.jsonl", "my/folder/folder1/", True),
            (PATHS, "/my/folder", "/folder1/folder2/*.jsonl", "my/folder/folder1/folder2/", True),
            (PATHS, "/my/folder", "/folder1/folder3/*.jsonl", "my/folder/folder1/folder3/", False),
            (PATHS, "/my/folder", "/folder1/folder3/*.csv", "my/folder/folder1/folder3/", False),
            (PATHS, "/my/folder", "/folder1/folder4/*.jsonl", "my/folder/folder1/folder4/", False),
            (PATHS, "/my/folder", "/folder4/*.jsonl", "my/folder/folder4/", False),
            (PATHS, "/my/folder", "/folder4/*/*.jsonl", "my/folder/folder4/", False),
            (PATHS, "/my/folder", "/*1/*.jsonl", "my/folder/", True),
            (PATHS, "/my/folder", "/folder1/f?ld*2/*.jsonl", "my/folder/folder1/", True),  # both ? and * wildcard
            # store url points to container root folder
            (["file1.jsonl"], "", "*.jsonl", "", True),
            (["file1.jsonl"], "", "*.csv", "", False),
            (["folder1/file1.jsonl", "folder1"], "", "*/*.jsonl", "", True),
            (["folder1/file1.jsonl", "folder1"], "", "folder1/*.jsonl", "folder1/", True),
            (["folder1/file1.jsonl", "folder1"], "", "*/*.csv", "", False),
        ]
    )
    def test_any_files(self, path_names, base_folder, pattern, expected_folder, expected_result):
        """Test any_files()."""
        def construct_mock_blob(name):
            mock_blob = Mock(is_directory='.' not in name)
            mock_blob.name = name
            return mock_blob

        def list_blobs(name_starts_with):
            assert name_starts_with == expected_folder, \
                f"expected non wildcard path to be {expected_folder}, but {name_starts_with} is given"
            # if condition to return only folders/files under the sub folder
            return [construct_mock_blob(name) for name in path_names if name.startswith(expected_folder)]

        def get_directory_client(folder):
            mock_dir_client = Mock(spec=DataLakeDirectoryClient)
            mock_dir_client.exists.return_value = any(name.startswith(folder) for name in path_names)
            return mock_dir_client

        def get_paths(path, recursive: bool):
            assert path == expected_folder, f"expected non wildcard path to be {expected_folder}, but {path} is given"
            assert recursive, "get_paths() should be called with recursive=True"
            return [construct_mock_blob(name) for name in path_names if name.startswith(expected_folder)]

        for store_type in ["blob", "gen2"]:
            if store_type == "blob":
                base_url = f"wasbs://my_container@my_account.blob.core.windows.net{base_folder}"
                mock_container_client = Mock(spec=ContainerClient)
                mock_container_client.list_blobs.side_effect = list_blobs
            else:
                base_url = f"abfss://my_container@my_account.dfs.core.windows.net{base_folder}"
                mock_container_client = Mock(spec=FileSystemClient)
                mock_container_client.get_directory_client.side_effect = get_directory_client
                mock_container_client.get_paths.side_effect = get_paths
            store_url = StoreUrl(base_url)
            assert store_url.any_files(pattern, mock_container_client) is expected_result

    @pytest.mark.parametrize(
        "pattern, expected_result",
        [
            ("2023/10/11/20/*.jsonl", True),
            ("2023/10/11/*/*.jsonl", True),
            ("2023/10/*/*/*.jsonl", True),
            ("2023/*/*/*/*.jsonl", True),
            ("2023/10/11/*/*.csv", False),
            ("2023/10/12/*/*.jsonl", False),
            ("2023/12/*/*/*.jsonl", False),
            ("2023/10/1?/*/m??_*_only.jsonl", True),
            ("2023/10/?2/*/*.jsonl", False)
        ]
    )
    def test_any_files_local(self, pattern, expected_result):
        """Test any_files() for local file system."""
        base_path = os.path.abspath(f"{os.path.dirname(__file__)}/raw_mdc_data")
        store_url = StoreUrl(base_path)
        assert store_url.any_files(pattern) is expected_result


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
    elif isinstance(credential1, AzureMLOnBehalfOfCredential):
        assert isinstance(credential2, AzureMLOnBehalfOfCredential)
        assert credential1._credential.get_client() == credential2._credential.get_client()
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
        assert_container_clients_properties_are_equal()
    else:
        raise NotImplementedError(f"Unsupported container client type: {type(container_client1)}")
