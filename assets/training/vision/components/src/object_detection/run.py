# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Run object detection task."""


from azureml.automl.core.shared.constants import Tasks
from azureml.automl.dnn.vision.common.constants import SettingsLiterals
from azureml.automl.dnn.vision.object_detection import runner
from azureml.core import Run

from common import utils
from common.settings import ObjectDetectionSettings


@utils.create_component_telemetry_wrapper(Tasks.IMAGE_OBJECT_DETECTION)
def run():
    """Run object detection task."""
    component_settings = ObjectDetectionSettings.create_from_parsing_current_cmd_line_args()
    mltable_data_json = utils.create_mltable_json(component_settings)
    runner.run(
        {SettingsLiterals.TASK_TYPE: Tasks.IMAGE_OBJECT_DETECTION},
        mltable_data_json=mltable_data_json)
    run = Run.get_context()
    utils.download_models(run, component_settings.mlflow_model_output, component_settings.pytorch_model_output)


if __name__ == "__main__":
    """Check GPU compute and run component."""
    utils.validate_running_on_gpu_compute()

    # Run the component.
    # (If multiple processes are spawned on the same node, only run the component on one process
    # since AutoML will spawn child processes as appropriate.)
    if utils.get_local_rank() == 0:
        run()
