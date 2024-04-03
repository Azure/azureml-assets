import numpy as np

from aggregator.base_aggregator import Aggregator

class MeanAggregator(Aggregator):

    def aggregate(self, data):
        return sum(data.get_values(self.config.identifier)) / len(data.get_values(self.config.identifier))
