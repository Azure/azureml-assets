# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wrapper code for running AutoML Many Models inference setup run."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.many_models.mm_setup_wrapper import (
    MMSetupWrapper
)


wrapper = MMSetupWrapper(is_train=False)
wrapper.run()
