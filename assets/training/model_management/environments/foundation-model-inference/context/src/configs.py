# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from dataclasses import dataclass
from constants import TaskType
from typing import Dict
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Dict, Type, TypeVar, Optional


@dataclass
class SerializableDataClass:
    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[TypeVar("T")], d: Dict) -> TypeVar("T"):
        return cls(**d)


@dataclass
class MiiEngineConfig(SerializableDataClass):
    deployment_name: str
    mii_configs: dict
    ds_config: Optional[str] = None
    ds_optimize: bool = True
    ds_zero: bool = False

    def __eq__(self, other):
        return (
            self.deployment_name == other.deployment_name
            and self.mii_configs == other.mii_configs
            and self.ds_config == other.ds_config
            and self.ds_optimize == other.ds_optimize
            and self.ds_zero == other.ds_zero
        )


@dataclass
class EngineConfig(SerializableDataClass):
    engine_name: str
    model_id: str
    tensor_parallel: int = 1
    trust_remote_code: bool = True
    mii_config: Optional[MiiEngineConfig] = None
    vllm_config: Optional[Dict] = None

    @classmethod
    def from_dict(cls: Type[TypeVar("T")], d: Dict) -> TypeVar("T"):
        if "mii_config" in d and isinstance(d["mii_config"], dict):
            d["mii_config"] = MiiEngineConfig.from_dict(d["mii_config"])
        return super().from_dict(d)

    def __eq__(self, other):
        for f in self.__dict__:
            if not f.startswith("_"):
                self_val = getattr(self, f)
                other_val = getattr(other, f)
                if self_val != other_val:
                    # print(f"Attribute {f} not equal: {self_val} != {other_val}")
                    return False
        return True


@dataclass
class TaskConfig(SerializableDataClass):
    task_type: TaskType = TaskType.TEXT_GENERATION

    def __eq__(self, other):
        return self.task_type == other.task_type
