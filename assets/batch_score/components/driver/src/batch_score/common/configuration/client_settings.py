from abc import ABC, abstractmethod
from enum import Enum


# To understand why we inherit from str,
# Question: https://stackoverflow.com/questions/58608361/string-based-enum-in-python
# And answer: https://stackoverflow.com/a/58608362
class ClientSettingsKey(str, Enum):
    # worker.py
    COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME = 'COUNT_ONLY_QUOTA_429_TOWARD_TOTAL_REQUEST_WAIT_TIME'
    NO_DEPLOYMENTS_BACK_OFF = 'NO_DEPLOYMENTS_BACK_OFF'

    # congestion.py
    SATURATION_THRESHOLD_P90_WAIT_TIME = 'SATURATION_THRESHOLD_P90_WAIT_TIME'
    CONGESTION_THRESHOLD_P90_WAIT_TIME = 'CONGESTION_THRESHOLD_P90_WAIT_TIME'

    # adjustment.py
    CONCURRENCY_ADJUSTMENT_INTERVAL = 'CONCURRENCY_ADJUSTMENT_INTERVAL'
    CONCURRENCY_ADDITIVE_INCREASE = 'CONCURRENCY_ADDITIVE_INCREASE'
    CONCURRENCY_MULTIPLICATIVE_DECREASE = 'CONCURRENCY_MULTIPLICATIVE_DECREASE'


class ClientSettingsProvider(ABC):
    @abstractmethod
    def get_client_setting(self, key: ClientSettingsKey) -> str:
        '''Returns the value of the client setting if it exists. Otherwise returns None.'''
        pass


class NullClientSettingsProvider(ClientSettingsProvider):
    def get_client_setting(self, key: ClientSettingsKey) -> str:
        return None
