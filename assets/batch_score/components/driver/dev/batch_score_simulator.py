# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Batch score simulator."""

import mltable
import os

import src.batch_score.main as main

from pathlib import Path

from endpoint_simulator import EndpointSimulator
from quota_simulator import QuotaSimulator
from routing_simulator import RoutingSimulator


class MiniBatchContext(object):
    """This is a context class containing partition and dataset info of mini-batches partitioned by keys."""

    def __init__(self, partition_key_value=None, dataset=None, minibatch_index=None):
        """Init the instance."""
        self._partition_key_value = partition_key_value
        self._dataset = dataset
        self._minibatch_index = minibatch_index

    @property
    def partition_key_value(self):
        """Return the dict of partition-key-value corresponding to the mini-batch."""
        return self._partition_key_value

    @property
    def dataset(self):
        """Return the sub dataset corresponding to the mini-batch."""
        return self._dataset

    @property
    def minibatch_index(self):
        """Return the minibatch identity."""
        return self._minibatch_index


# Simulate PRS with a single Processor on a single Node
class Simulator:
    """PRS Simulator."""

    def __init__(self, data_input_folder_path):
        """Init function."""
        self.__mltable_data: mltable = mltable.load(data_input_folder_path)
        self.__df_data = self.__mltable_data.to_pandas_dataframe()
        self.__minibatch_size = 500  # lines
        self.__cur_index = 0

    def start(self):
        """Start the simulator."""
        main.init()
        results: list[str] = []

        while self.__cur_index < self.__df_data.shape[0]:
            end_index = self.__cur_index + self.__minibatch_size
            if end_index > self.__df_data.shape[0]:
                end_index = self.__df_data.shape[0]
            df_subset = self.__df_data.iloc[self.__cur_index:end_index]
            self.__cur_index = end_index

            results.extend(main.run(df_subset, MiniBatchContext(minibatch_index=10)))

        main.shutdown()

        out_dir = "./out"
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(
            os.path.join(out_dir, "prs-sim.txt"), "wt", encoding="utf-8"
        ) as txt_file:
            print("\n".join(results), file=txt_file)


EndpointSimulator.initialize()
QuotaSimulator.initialize()
RoutingSimulator.initialize()

sim = Simulator("./training/")
sim.start()
