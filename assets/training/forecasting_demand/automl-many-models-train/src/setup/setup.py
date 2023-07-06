# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wrapper code for running AutoML Many Models train setup run."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.many_models.mm_setup_wrapper import (
    MMSetupWrapper
)


wrapper = MMSetupWrapper()
wrapper.run()
