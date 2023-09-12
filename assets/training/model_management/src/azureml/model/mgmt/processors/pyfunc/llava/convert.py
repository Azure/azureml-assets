import os
import shutil
import sys

import mlflow

from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema

from llava_model_wrapper import LLaVAMLflowWrapper, MLflowLiterals


def _get_mlflow_signature(task_type: str) -> ModelSignature:
    """Return MLflow model signature with input and output schema given the input task type.
    :param task_type: Task type used in training
    :type task_type: str
    :return: MLflow model signature.
    :rtype: mlflow.models.signature.ModelSignature
    """

    # input_schema = Schema(
    #     [ColSpec(MLflowSchemaLiterals.STRING_DATA_TYPE, ColumnNames.TEXT_PROMPT)]
    # )

    # output_schema = Schema(
    #     [ColSpec(MLflowSchemaLiterals.STRING_DATA_TYPE, ColumnNames.TEXT_PROMPT),
    #      ColSpec(MLflowSchemaLiterals.IMAGE_DATA_TYPE, ColumnNames.GENERATED_IMAGE),
    #      ColSpec(MLflowSchemaLiterals.STRING_DATA_TYPE, ColumnNames.NSFW_FLAG),
    #     ]
    # )

    # return ModelSignature(inputs=input_schema, outputs=output_schema)

    return None


if __name__ == "__main__":
    root_dir = "C:\\Source\\azureml-assets\\assets\\training\\model_management\\src\\azureml\\model\\mgmt\\processors\\pyfunc\\llava"
    input_dir = root_dir + "\\inp"
    output_dir = root_dir + "\\outp"

    # This to get Wrapper class independently and not as part of parent package.
    sys.path.append(os.path.dirname(__file__))

    mlflow_model_wrapper = LLaVAMLflowWrapper(task_type="", model_id="7b")
    # pip_requirements = os.path.join(os.path.dirname(__file__), "requirements.txt")
    code_path = [
        os.path.join(os.path.dirname(__file__), "llava_model_wrapper.py"),
        os.path.join(os.path.dirname(__file__), "constants.py"),
    ]

    model_dir = os.path.join(os.path.dirname(input_dir), "model_dir")
    shutil.copytree(input_dir, model_dir, dirs_exist_ok=True)

    mlflow.pyfunc.save_model(
        path=output_dir,
        python_model=mlflow_model_wrapper,
        artifacts={MLflowLiterals.MODEL_DIR: model_dir},
        # signature=_get_mlflow_signature("42"),
        # pip_requirements=pip_requirements,
        # code_path=code_path,
        # metadata={MLflowLiterals.MODEL_NAME: model_id},
    )
