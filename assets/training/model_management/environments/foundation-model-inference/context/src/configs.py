# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Configuration classes for the Engine and Task."""
from dataclasses import asdict, dataclass, field
from typing import Dict, Type, TypeVar, Optional

from constants import TaskType


@dataclass
class SerializableDataClass:
    """A data class that can be serialized to and from a dictionary."""

    def to_dict(self) -> Dict:
        """Convert the data class to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls: Type[TypeVar("T")], d: Dict) -> TypeVar("T"):
        """Create a data class from a dictionary."""
        return cls(**d)


@dataclass
class MiiEngineConfig(SerializableDataClass):
    """Configuration for the Mii Engine."""

    deployment_name: str
    mii_configs: dict
    ds_config: dict = field(default_factory=dict)
    enable_deepspeed: bool = True
    ds_zero: bool = False

    def __eq__(self, other):
        """Check if this configuration is equal to another."""
        return (
            self.deployment_name == other.deployment_name
            and self.mii_configs == other.mii_configs
            and self.ds_config == other.ds_config
            and self.enable_deepspeed == other.enable_deepspeed
            and self.ds_zero == other.ds_zero
        )


@dataclass
class EngineConfig(SerializableDataClass):
    """Configuration for the Engine."""

    engine_name: str
    model_id: str
    tensor_parallel: Optional[int] = None
    trust_remote_code: bool = True
    mii_config: Optional[MiiEngineConfig] = None
    vllm_config: Optional[Dict] = None

    @classmethod
    def from_dict(cls: Type[TypeVar("T")], d: Dict) -> TypeVar("T"):
        """Create a configuration from a dictionary."""
        if "mii_config" in d and isinstance(d["mii_config"], dict):
            d["mii_config"] = MiiEngineConfig.from_dict(d["mii_config"])
        return super().from_dict(d)

    def __eq__(self, other):
        """Check if this configuration is equal to another."""
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
    """Configuration for the Task."""

    task_type: TaskType = TaskType.TEXT_GENERATION

    def __eq__(self, other):
        """Check if this configuration is equal to another."""
        return self.task_type == other.task_type
