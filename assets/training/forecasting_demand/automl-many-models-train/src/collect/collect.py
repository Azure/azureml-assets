# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wrapper code for running AutoML Many Models train collect run."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.many_models.mm_collect_wrapper import (
    MMCollectWrapper
)


wrapper = MMCollectWrapper()
wrapper.run()
