import json

from graders.config.equality_grader_config import EqualityGraderConfig


def get_config(identifier, config_dict):
    if identifier == "Equality":
        return EqualityGraderConfig(identifier=identifier, **config_dict)
    raise ValueError(f"Unknown identifier: {identifier}")