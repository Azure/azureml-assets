# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Wrapper for HTS inference setup."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.hts.hts_setup_wrapper import (
    HTSSetupWrapper
)


wrapper = HTSSetupWrapper(is_train=False)
wrapper.run()
