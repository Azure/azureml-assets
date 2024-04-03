from aggregator.mean_aggregator import MeanAggregator


def get_aggregator(identifier):
    if identifier == "Equality":
        return MeanAggregator
    raise ValueError(f"Unknown identifier: {identifier}")
