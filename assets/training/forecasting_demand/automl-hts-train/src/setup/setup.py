# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Setup wrapper for HTS train."""
from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.hts.hts_setup_wrapper import (
    HTSSetupWrapper
)


wrapper = HTSSetupWrapper()
wrapper.run()
