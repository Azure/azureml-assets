from dataclasses import dataclass

from graders.config.base_config import BaseConfig
from graders.equality_garder import EqualityGrader


@dataclass
class EqualityGraderConfig(BaseConfig):
    type: str
    element: str
    caseInsensitive: bool
    presentValue: float
    absentValue: float

