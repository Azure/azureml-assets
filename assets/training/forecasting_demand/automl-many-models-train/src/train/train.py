# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------
"""Wrapper code for running AutoML Many Models PRS training run."""
from pathlib import Path
import os


# Clearing this environment variable avoids periodic calls from
# dprep log uploading to Run.get_context() and cause RH throttling
# when running at scale. It looks like this logging path repeatedly uploads timespan
# tracing data to the PRS step itself from each worker.
os.environ["AZUREML_OTEL_EXPORT_RH"] = ""

# Batch / flush metrics in the many models scenario
os.environ["AZUREML_METRICS_POLLING_INTERVAL"] = '30'

# Once the metrics service has uploaded & queued metrics for processing, we don't
# need to wait for those metrics to be ingested on flush.
os.environ['AZUREML_FLUSH_INGEST_WAIT'] = ''


from azureml.train.automl.runtime._solution_accelorators.pipeline_run.steps.many_models.mm_automl_train_v2 import \
    MMAutoMLTrainWrapperV2   # noqa: E402


# Whether or not this driver has been initialized
driver_initialized = False
runtime_wrapper = None


def run(mini_batches):
    """Run method as PRS required."""
    # Initiailze the driver when the first mini batch runs.
    # (This is done because the initialize call requires a sleep call to stagger new traffic ramp-up (refer to
    # the initialize method for more info). There can be a large time gap between calls to the PRS in-built init()
    # methods and the in-built run methods. For example, for large datasets, it seems possible for PRS to invoke
    # the init() methods of all workers, and then 15 minutes later, invoke the run methods of all workers. Given that,
    # the sleep call to stagger traffic ramp-up won't work as expected if invoked in the PRS in-built init() method.)
    global runtime_wrapper
    global driver_initialized

    if runtime_wrapper is None:
        runtime_wrapper = MMAutoMLTrainWrapperV2(Path(__file__).parent.absolute())
        runtime_wrapper.init_prs()
    result_list = runtime_wrapper.run_prs(mini_batches)
    print("AutoML train step init is done.")
    return result_list
