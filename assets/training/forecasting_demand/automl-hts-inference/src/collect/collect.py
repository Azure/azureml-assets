# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wrapper for HTS inferece collect."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.hts.hts_collect_wrapper import (
    HTSCollectWrapper
)


wrapper = HTSCollectWrapper(is_train=False)
wrapper.run()
