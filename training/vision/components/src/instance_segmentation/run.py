import argparse
import json

from azureml.automl.core.shared.constants import MLTableDataLabel, MLTableLiterals, Tasks
from azureml.automl.dnn.vision.common import utils
from azureml.automl.dnn.vision.common.constants import SettingsLiterals
from azureml.automl.dnn.vision.object_detection import runner
from azureml.core import Run
from common.telemetry_utils import create_component_telemetry_wrapper


@create_component_telemetry_wrapper(Tasks.IMAGE_INSTANCE_SEGMENTATION)
def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(utils._make_arg('training_data'), type=str)
    parser.add_argument(utils._make_arg('validation_data'), type=str)
    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    mltable_data_dict = {
        MLTableDataLabel.TrainData.value: {
            MLTableLiterals.MLTABLE_RESOLVEDURI: args_dict['training_data']
        }
    }

    if args_dict['validation_data']:
        mltable_data_dict[MLTableDataLabel.ValidData.value] = {
            MLTableLiterals.MLTABLE_RESOLVEDURI: args_dict['validation_data']
        }

    mltable_data_json = json.dumps(mltable_data_dict)

    settings = {SettingsLiterals.TASK_TYPE: Tasks.IMAGE_INSTANCE_SEGMENTATION}
    runner.run(settings, mltable_data_json=mltable_data_json)
    run = Run.get_context()
    run.download_files(prefix='outputs/mlflow-model', output_directory=args_dict['model_output'], append_prefix=False)


if __name__ == "__main__":
    run()
