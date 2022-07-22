# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from azureml.automl.core.shared.constants import Tasks
from azureml.automl.dnn.vision.common.constants import SettingsLiterals
from azureml.automl.dnn.vision.object_detection import runner
from azureml.core import Run

from common import utils
from common.settings import InstanceSegmentationSettings


@utils.create_component_telemetry_wrapper(Tasks.IMAGE_INSTANCE_SEGMENTATION)
def run():
    component_settings = InstanceSegmentationSettings.create_from_parsing_current_cmd_line_args()
    mltable_data_json = utils.create_mltable_json(component_settings)
    runner.run(
        {SettingsLiterals.TASK_TYPE: Tasks.IMAGE_INSTANCE_SEGMENTATION},
        mltable_data_json=mltable_data_json)
    run = Run.get_context()
    run.download_files(
        prefix='outputs/mlflow-model', output_directory=component_settings.model_output, append_prefix=False)


if __name__ == "__main__":
    utils.validate_running_on_gpu_compute()

    # Run the component.
    # (If multiple processes are spawned on the same node, only run the component on one process
    # since AutoML will spawn child processes as appropriate.)
    if utils.get_local_rank() == 0:
        run()
