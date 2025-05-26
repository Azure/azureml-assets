# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run an image classification task with specified component settings."""


from azureml.automl.core.shared.constants import Tasks
from azureml.automl.dnn.vision.common.constants import SettingsLiterals
from azureml.automl.dnn.vision.classification import runner
from azureml.core import Run

from common import utils
from common.settings import ClassificationSettings


def run(task, component_settings):
    """Run task with given component settings."""
    @utils.create_component_telemetry_wrapper(task)
    def run_component(task):
        mltable_data_json = utils.create_mltable_json(component_settings)
        runner.run(
            {SettingsLiterals.TASK_TYPE: task},
            mltable_data_json=mltable_data_json, multilabel=component_settings.multilabel)
        run = Run.get_context()
        utils.download_models(run, component_settings.mlflow_model_output, component_settings.pytorch_model_output)

    run_component(task)


if __name__ == "__main__":
    """Check GPU compute and run component."""
    utils.validate_running_on_gpu_compute()

    # Run the component.
    # (If multiple processes are spawned on the same node, only run the component on one process
    # since AutoML will spawn child processes as appropriate.)
    if utils.get_local_rank() == 0:
        component_settings = ClassificationSettings.create_from_parsing_current_cmd_line_args()
        task = Tasks.IMAGE_CLASSIFICATION_MULTILABEL if component_settings.multilabel else \
            Tasks.IMAGE_CLASSIFICATION
        run(task, component_settings)
