# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Entry script for Hello world Component."""

from shared_utilities.io_utils import init_spark
from shared_utilities import constants


def run():
    """Hello world."""
    spark = init_spark()
    nums = spark.sparkContext.parallelize([1,2,3,4])
    print(nums.map(lambda x: x*x).collect())


if __name__ == "__main__":

    run()
