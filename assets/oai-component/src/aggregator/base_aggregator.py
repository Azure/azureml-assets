from abc import ABC, abstractmethod

from aggregator.config.base_config import AggregatorConfig
from dataclasses import dataclass


@dataclass
class Aggregator(ABC):
    config: AggregatorConfig

    @abstractmethod
    def aggregate(self, data):
        pass
