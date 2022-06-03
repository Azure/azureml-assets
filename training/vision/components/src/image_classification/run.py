import argparse
import json

from azureml.automl.core.shared.constants import MLTableDataLabel, MLTableLiterals, Tasks
from azureml.automl.dnn.vision.common import utils
from azureml.automl.dnn.vision.common.constants import SettingsLiterals
from azureml.automl.dnn.vision.classification import runner
from common.telemetry_utils import create_component_telemetry_wrapper
from azureml.core import Run

def runClassificationComponent(task_type):
    @create_component_telemetry_wrapper(task_type)
    def runnerComponent():
        parser = argparse.ArgumentParser()
        parser.add_argument(utils._make_arg('training_data'), type=str)
        parser.add_argument(utils._make_arg('validation_data'), type=str)
        parser.add_argument(utils._make_arg('model_output'), type=str)
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

        settings = {SettingsLiterals.TASK_TYPE: task_type}
        runner.run(settings, mltable_data_json=mltable_data_json)

        run = Run.get_context()
        run.download_files(prefix='outputs/mlflow-model', output_directory=args_dict['model_output'], append_prefix=False)
    
    runnerComponent()


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument(utils._make_arg('multilabel'), type=bool)
    args, _ = parser.parse_known_args()
    args_dict = vars(args)

    task_type= Tasks.IMAGE_CLASSIFICATION

    if args_dict['multilabel']== True:
        task_type = Tasks.IMAGE_CLASSIFICATION_MULTILABEL

    runClassificationComponent(task_type)

if __name__ == "__main__":
    run()
