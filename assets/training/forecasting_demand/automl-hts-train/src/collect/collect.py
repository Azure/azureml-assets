# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Collect wrapper for HTS train."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.hts.hts_collect_wrapper import (
    HTSCollectWrapper
)


wrapper = HTSCollectWrapper()
wrapper.run()
