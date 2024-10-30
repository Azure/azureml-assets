# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests running in mlflow 20.04 py39 cpu environment."""
import sys
import os
import json
import pytest
import shutil
import yaml
import pandas as pd
import numpy as np
from mlflow.models.signature import ModelSignature
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
sys.path.append(os.path.abspath("../context"))


@pytest.fixture(scope="session")
def teardown():
    """For Teardown."""
    yield
    try:
        shutil.rmtree("mlruns")
    except:  # noqa: E722
        pass


def setup_traditional():
    """For Setup traditional."""
    import mlflow

    # define a custom model
    class MyModel(mlflow.pyfunc.PythonModel):
        def predict(self, context, model_input):
            return self.my_custom_function(model_input)

        def my_custom_function(self, model_input):
            # do something with the model input
            return 0

    # save the model
    my_model = MyModel()
    # if you have an error about YAML, delete mlruns directory
    with mlflow.start_run():
        from pathlib import Path
        import posixpath
        path = os.path.join(Path.cwd(), "./resources/mlflow_unit/dummy/")
        path = posixpath.normpath(path)
        # fix for running on unix
        if path.startswith("/"):
            path = "~" + path
        mlflow.pyfunc.log_model(artifact_path=path, python_model=my_model)
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "dummy"
    try:
        del sys.modules['mlflow_score_script']
    except:  # noqa: E722
        pass
    try:
        del sys.modules['inference_schema']
    except:  # noqa: E722
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()


def setup_transformers():
    """For Setup transformers."""
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "distilbert-base-uncased"
    try:
        del sys.modules['mlflow_score_script']
    except:  # noqa: E722
        pass
    try:
        del sys.modules['inference_schema']
    except:  # noqa: E722
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()

def test_parse_model_input_from_input_data_dict_to_pandas(teardown):
    """Test parse model input from input data dict to pandas."""
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_traditional

    with open("./resources/mlflow/sample_2_0_input.txt", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = parse_model_input_from_input_data_traditional(input_data1)
        assert isinstance(result1, pd.DataFrame)
        assert len(result1) == 1
        assert result1.loc[0].at["a"] == 3.0
        assert result1.loc[0].at["c"] == "foo"

def test_get_sample_input_from_loaded_example_pandas_2_0(teardown):
    """Test get sample input from loaded example pandas 2 0."""
    setup_traditional()
    from mlflow_score_script import get_sample_input_from_loaded_example

    example_info = {
        "type": "dataframe",
        "pandas_orient": "split"
    }

    with open("./resources/mlflow/mlflow_2_0_model_folder/input_example.json", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = get_sample_input_from_loaded_example(example_info, input_data1)
        assert isinstance(result1, pd.DataFrame)
        assert result1.loc[0].at["a"] == 3.0
        assert result1.loc[0].at["c"] == "foo"

def test_get_sample_input_from_loaded_example_str_pandas(teardown):
    """Test get sample input from loaded example str pandas."""
    setup_traditional()
    from mlflow_score_script import get_sample_input_from_loaded_example

    example_info = {
        "type": "dataframe",
        "pandas_orient": "split"
    }

    with open("./resources/mlflow_unit/str_translation/input_example.json", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = get_sample_input_from_loaded_example(example_info, input_data1)
        assert isinstance(result1, pd.DataFrame)
        # TODO should it be wrapped in a list?
        assert result1.loc[0].at["data"] == ["MLflow is great!"]


def test_get_sample_input_from_loaded_example_dict_str_pandas(teardown):
    """Test get sample input from loaded example dict str pandas."""
    setup_traditional()
    from mlflow_score_script import get_sample_input_from_loaded_example

    example_info = {
        "type": "dataframe",
        "pandas_orient": "split"
    }

    with open("./resources/mlflow_unit/dict_str_qa/input_example.json", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = get_sample_input_from_loaded_example(example_info, input_data1)
        assert isinstance(result1, pd.DataFrame)
        assert result1.loc[0].at["question"] == "Why is model conversion important?"


def test_get_samples_from_signature_2_0_pandas(teardown):
    """Test get samples from signature 2 0 pandas."""
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow/mlflow_2_0_model_folder/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        assert isinstance(result1, pd.DataFrame)
        assert result1.dtypes[0] == float
        assert result1.dtypes[2] == np.dtype('O')


def test_get_samples_from_signature_pandas(teardown):
    """Test get samples from signature pandas."""
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow/mlflow_model_folder/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        assert isinstance(result1, pd.DataFrame)
        assert result1.dtypes[0] == np.dtype('int64')
        assert result1.dtypes[1] == np.dtype('int64')