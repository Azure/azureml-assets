from aggregator.config.base_config import AggregatorConfig


def get_config(identifier, config_dict):
    if identifier == "Equality":
        return AggregatorConfig(identifier=identifier)
    raise ValueError(f"Unknown identifier: {identifier}")
