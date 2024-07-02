from abc import ABC, abstractmethod
from dataclasses import dataclass

from graders.config.base_config import BaseConfig


@dataclass
class Grader(ABC):
    config: BaseConfig

    @abstractmethod
    def compute(self, preds, ground_truth):
        pass
