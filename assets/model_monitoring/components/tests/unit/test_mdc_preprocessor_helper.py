# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test file for MDC preprocessor helper."""

from unittest.mock import Mock
import pytest
from datetime import datetime, timezone
from azure.identity import ClientSecretCredential, DefaultAzureCredential
from azure.core.credentials import AzureSasCredential
from model_data_collector_preprocessor.mdc_preprocessor_helper import (
    get_file_list, set_data_access_config, serialize_credential, deserialize_credential
)
from src.shared_utilities.store_url import StoreUrl
from test_store_url import assert_credentials_are_equal


class MyDateTime:
    """helper class for test_get_file_list()."""

    def __init__(self, year, month=None, day=None, hour=None):
        """Initialize MyDateTime, month, day and hour are all optional."""
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour

    def abfs_path(self):
        """Return the abfs path of this MyDateTime."""
        month, day, hour = self._get_month_day_hour()
        return f"abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/{self.year}/{month}/{day}/{hour}/*.jsonl"  # noqa

    def azureml_path(self):
        """Return the azureml path of this MyDateTime."""
        month, day, hour = self._get_month_day_hour()
        return f"azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_workspace/datastores/my_datastore/path/to/folder/{self.year}/{month}/{day}/{hour}/*.jsonl"  # noqa

    def _get_month_day_hour(self):
        """Return month, day, hour."""
        month = f"{self.month:02d}" if self.month else "*"
        day = f"{self.day:02d}" if self.day else "*"
        hour = f"{self.hour:02d}" if self.hour is not None else "*"
        return month, day, hour


@pytest.mark.unit
class TestMDCPreprocessorHelper:
    """Test class for MDC preprocessor helper."""

    NONE_EMPTY_DATETIMES = [
        datetime(2020, 5, 6), datetime(2020, 9, 18),
        datetime(2022, 9, 17), datetime(2022, 11, 15),
        datetime(2023, 10, 12, 2), datetime(2023, 10, 12, 15),
        datetime(2023, 10, 20, 11), datetime(2023, 10, 20, 20),
        datetime(2023, 11, 19, 5), datetime(2023, 11, 19, 16), datetime(2023, 11, 19, 22),
        datetime(2023, 11, 20, 6), datetime(2023, 11, 20, 7), datetime(2023, 11, 20, 8), datetime(2023, 11, 20, 13),
        datetime(2023, 11, 20, 17), datetime(2023, 11, 20, 19), datetime(2023, 11, 20, 20), datetime(2023, 11, 20, 21),
        datetime(2023, 11, 21, 3), datetime(2023, 11, 21, 9), datetime(2023, 11, 21, 14), datetime(2023, 11, 21, 23),
        datetime(2023, 11, 23), datetime(2023, 11, 23, 9),
        datetime(2023, 12, 1), datetime(2023, 12, 1, 10),
        datetime(2023, 12, 10, 11), datetime(2023, 12, 10, 22),
        datetime(2024, 1, 1), datetime(2024, 1, 1, 10),
        datetime(2024, 1, 12, 16), datetime(2024, 1, 12, 21),
        datetime(2024, 2, 3, 3), datetime(2024, 2, 3, 9),
    ]

    MATCH_PATTERNS = [
        p for d in NONE_EMPTY_DATETIMES
        for p in (
            f"{d.strftime('%Y/%m/%d/%H')}/*.jsonl",
            f"{d.strftime('%Y/%m/%d')}/*/*.jsonl",
            f"{d.strftime('%Y/%m')}/*/*/*.jsonl",
            f"{d.strftime('%Y')}/*/*/*/*.jsonl"
        )
    ]

    @pytest.mark.parametrize(
        "start, end, expected_datetimes",
        [
            # same day
            (
                datetime(2023, 11, 20, 7), datetime(2023, 11, 20, 21),
                [
                    MyDateTime(2023, 11, 20, 7), MyDateTime(2023, 11, 20, 8), MyDateTime(2023, 11, 20, 13),
                    MyDateTime(2023, 11, 20, 17), MyDateTime(2023, 11, 20, 19), MyDateTime(2023, 11, 20, 20)
                ]
            ),
            (
                datetime(2023, 11, 20, 4), datetime(2023, 11, 20, 8),
                [MyDateTime(2023, 11, 20, 6), MyDateTime(2023, 11, 20, 7)]
            ),
            (
                datetime(2023, 11, 20, 20), datetime(2023, 11, 20, 23),
                [MyDateTime(2023, 11, 20, 20), MyDateTime(2023, 11, 20, 21)]
            ),
            (
                datetime(2023, 11, 20, 13), datetime(2023, 11, 20, 19),
                [MyDateTime(2023, 11, 20, 13), MyDateTime(2023, 11, 20, 17)]
            ),
            (datetime(2023, 11, 20, 7), datetime(2023, 11, 20, 8), [MyDateTime(2023, 11, 20, 7)]),
            (datetime(2023, 11, 20), datetime(2023, 11, 21), [MyDateTime(2023, 11, 20)]),  # whole day
            (datetime(2023, 11, 20, 3), datetime(2023, 11, 20, 6), []),
            (datetime(2023, 11, 20, 22), datetime(2023, 11, 20, 23), []),
            (datetime(2023, 11, 20, 9), datetime(2023, 11, 20, 13), []),
            # cross day
            (
                datetime(2023, 11, 20, 20), datetime(2023, 11, 21, 10),
                [
                    MyDateTime(2023, 11, 20, 20), MyDateTime(2023, 11, 20, 21),
                    MyDateTime(2023, 11, 21, 3), MyDateTime(2023, 11, 21, 9)
                ]
            ),
            (
                datetime(2023, 11, 19, 10), datetime(2023, 11, 21, 14),
                [
                    MyDateTime(2023, 11, 19, 16), MyDateTime(2023, 11, 19, 22),
                    MyDateTime(2023, 11, 20),
                    MyDateTime(2023, 11, 21, 3), MyDateTime(2023, 11, 21, 9)
                ]
            ),
            (
                datetime(2023, 11, 19, 16), datetime(2023, 11, 22, 13),
                [
                    MyDateTime(2023, 11, 19, 16), MyDateTime(2023, 11, 19, 22),
                    MyDateTime(2023, 11, 20), MyDateTime(2023, 11, 21)
                ]
            ),
            (
                datetime(2023, 11, 19, 17), datetime(2023, 11, 23, 8),
                [
                    MyDateTime(2023, 11, 19, 22),
                    MyDateTime(2023, 11, 20), MyDateTime(2023, 11, 21),
                    MyDateTime(2023, 11, 23, 0)
                ]
            ),
            (
                datetime(2023, 11, 19, 17), datetime(2023, 11, 26, 8),
                [
                    MyDateTime(2023, 11, 19, 22),
                    MyDateTime(2023, 11, 20), MyDateTime(2023, 11, 21), MyDateTime(2023, 11, 23)
                ]
            ),
            (datetime(2023, 11, 20), datetime(2023, 11, 23), [MyDateTime(2023, 11, 20), MyDateTime(2023, 11, 21)]),
            (datetime(2023, 11, 1), datetime(2023, 12, 1), [MyDateTime(2023, 11)]),  # whole month
            (datetime(2023, 11, 24, 9), datetime(2023, 11, 30, 12), []),
            # cross month
            (
                datetime(2023, 11, 21, 14), datetime(2023, 12, 10, 12),
                [
                    MyDateTime(2023, 11, 21, 14), MyDateTime(2023, 11, 21, 23),
                    MyDateTime(2023, 11, 23), MyDateTime(2023, 12, 1),
                    MyDateTime(2023, 12, 10, 11)
                ]
            ),
            (
                datetime(2023, 10, 12, 3), datetime(2023, 12, 10, 23, 50),
                [
                    MyDateTime(2023, 10, 12, 15), MyDateTime(2023, 10, 20),
                    MyDateTime(2023, 11),
                    MyDateTime(2023, 12, 1), MyDateTime(2023, 12, 10)
                ]
            ),
            (
                datetime(2023, 10, 12, 3), datetime(2023, 12, 10, 23),
                [
                    MyDateTime(2023, 10, 12, 15), MyDateTime(2023, 10, 20),
                    MyDateTime(2023, 11),
                    MyDateTime(2023, 12, 1), MyDateTime(2023, 12, 10, 11), MyDateTime(2023, 12, 10, 22)
                ]
            ),
            (datetime(2023, 8, 1), datetime(2023, 12, 1), [MyDateTime(2023, 10), MyDateTime(2023, 11)]),
            (datetime(2023, 1, 1), datetime(2024, 1, 1), [MyDateTime(2023)]),  # whole year
            (
                datetime(2023, 10, 1), datetime(2023, 12, 5),
                [MyDateTime(2023, 10), MyDateTime(2023, 11), MyDateTime(2023, 12, 1)]
            ),
            (
                datetime(2023, 10, 12), datetime(2024, 1, 1),
                [MyDateTime(2023, 10, 12), MyDateTime(2023, 10, 20), MyDateTime(2023, 11), MyDateTime(2023, 12)]
            ),
            # cross year
            (
                datetime(2023, 10, 12, 3), datetime(2024, 2, 10, 23),
                [
                    MyDateTime(2023, 10, 12, 15), MyDateTime(2023, 10, 20), MyDateTime(2023, 11), MyDateTime(2023, 12),
                    MyDateTime(2024, 1), MyDateTime(2024, 2, 3)
                ]
            ),
            (
                datetime(2022, 10, 12), datetime(2024, 2, 10),
                [
                    MyDateTime(2022, 11),
                    MyDateTime(2023),
                    MyDateTime(2024, 1), MyDateTime(2024, 2, 3)
                ]
            ),
            (datetime(2022, 1, 1), datetime(2024, 2, 1), [MyDateTime(2022), MyDateTime(2023), MyDateTime(2024, 1)]),
            (
                datetime(2022, 5, 1), datetime(2025, 1, 1),
                [MyDateTime(2022, 9), MyDateTime(2022, 11), MyDateTime(2023), MyDateTime(2024)]
            ),
            (
                datetime(2019, 5, 1), datetime(2025, 8, 1),
                [MyDateTime(2020), MyDateTime(2022), MyDateTime(2023), MyDateTime(2024)]
            ),
            (datetime(2020, 1, 1), datetime(2024, 1, 1), [MyDateTime(2020), MyDateTime(2022), MyDateTime(2023)]),
        ]
    )
    def test_get_file_list(self, start, end, expected_datetimes):
        """Test get_file_list()."""

        mock_store_url = Mock(spec=StoreUrl)
        mock_store_url.is_local_path.return_value = False
        mock_store_url.any_files.side_effect = lambda pattern: pattern in TestMDCPreprocessorHelper.MATCH_PATTERNS
        mock_store_url.get_abfs_url.side_effect = \
            lambda rpath: f"abfss://my_container@my_account.dfs.core.windows.net/path/to/folder/{rpath}"
        mock_store_url.get_azureml_url.side_effect = \
            lambda rpath: ("azureml://subscriptions/sub_id/resourceGroups/my_rg/workspaces/my_workspace"
                           f"/datastores/my_datastore/path/to/folder/{rpath}")

        for scheme in ["abfs", "azureml"]:
            expected_result = [d.abfs_path() if scheme == "abfs" else d.azureml_path() for d in expected_datetimes]
            for i in range(4):
                # test with and without timezone
                _start = start.replace(tzinfo=timezone.utc) if i % 2 else start
                _end = end.replace(tzinfo=timezone.utc) if (i >> 1) % 2 else end
                # scheme = "abfs" if (i >> 2) % 2 else "azureml"

                file_list = get_file_list(_start, _end, mock_store_url, scheme=scheme)

                assert file_list == expected_result

    @pytest.mark.parametrize(
        "is_local, store_type, credential, expected_spark_conf",
        [
            (True, None, None, {}),
            (False, "dfs", None, {}),
            (False, "blob", "account_key", {
                "fs.azure.account.auth.type.my_account.dfs.core.windows.net": "SharedKey",
                "fs.azure.account.key.my_account.dfs.core.windows.net": "account_key"
            }),
            (False, "blob", AzureSasCredential("sas_token"), {
                "fs.azure.account.auth.type.my_account.dfs.core.windows.net": "SAS",
                "fs.azure.sas.token.provider.type.my_account.dfs.core.windows.net":
                    "com.microsoft.azure.synapse.tokenlibrary.ConfBasedSASProvider",
                "spark.storage.synapse.my_container.my_account.dfs.core.windows.net.sas": "sas_token"
            }),
            (False, "blob", ClientSecretCredential("00000", "client_id", "client_secret"), {})
        ]
    )
    def test_set_data_access_config(self, is_local, store_type, credential, expected_spark_conf):
        """Test set_data_access_config()."""
        class MyDict(dict):
            def __init__(self, d: dict):
                super().__init__(d)

            def set(self, k, v):
                self[k] = v

        mock_spark = Mock(conf=MyDict({}))
        mock_store_url = Mock(spec=StoreUrl, store_type=store_type,
                              account_name="my_account", container_name="my_container")
        mock_store_url.is_local_path.return_value = is_local
        mock_store_url.get_credential.return_value = credential

        set_data_access_config(mock_spark, store_url=mock_store_url)

        assert mock_spark.conf == expected_spark_conf

    @pytest.mark.parametrize(
        "credential",
        [
            "account_key",
            AzureSasCredential("sas_token"),
            ClientSecretCredential("00000", "client_id", "client_secret"),
            DefaultAzureCredential(),
            None
        ]
    )
    def test_serialize_deserialize_credential(self, credential):
        """Test serialize_credential() and deserialize_credential()."""
        serialized_credential = serialize_credential(credential)
        deserialized_credential = deserialize_credential(serialized_credential)

        expected_credential = None if isinstance(credential, DefaultAzureCredential) else credential
        assert_credentials_are_equal(deserialized_credential, expected_credential)
