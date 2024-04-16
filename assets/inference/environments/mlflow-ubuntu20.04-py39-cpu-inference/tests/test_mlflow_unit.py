# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import sys, os, json, pytest, tempfile, shutil, yaml
import pandas as pd
import numpy as np
from mlflow.models.signature import ModelSignature
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
sys.path.append(os.path.abspath("../context"))


@pytest.fixture(scope="session")
def teardown():
    yield
    try:
        shutil.rmtree("mlruns")
    except:
        pass

def setup_traditional():
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
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()

def setup_transformers():
    import mlflow
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "distilbert-base-uncased"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()


def test_parse_model_input_from_input_data_dict_to_tensor(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_traditional

    input_data1 = '{"columns": ["a", "b", "c"], "data": [[3.0, 1, "foo"]]}'
    result1 = parse_model_input_from_input_data_traditional(input_data1)
    np.testing.assert_equal(result1, {"columns": np.asarray(["a", "b", "c"]), "data": np.asarray([[3.0, 1, "foo"]])})

    input_data2 = {"columns": ["a", "b", "c"], "data": [[3.0, 1, "foo"]]}
    result2 = parse_model_input_from_input_data_traditional(input_data2)
    np.testing.assert_equal(result2, {"columns": np.asarray(["a", "b", "c"]), "data": np.asarray([[3.0, 1, "foo"]])})

    input_data3 = '{"columns": ["a", "b"], "data": [[0, 1]]}'
    result3 = parse_model_input_from_input_data_traditional(input_data3)
    np.testing.assert_equal(result3, {"columns": np.asarray(["a", "b"]), "data": np.asarray([[0, 1]])})

    input_data4 = {"columns": ["a", "b"], "data": [[0, 1]]}
    result4 = parse_model_input_from_input_data_traditional(input_data4)
    np.testing.assert_equal(result4, {"columns": np.asarray(["a", "b"]), "data": np.asarray([[0, 1]])})

    with open("./resources/mlflow/named_tensor_input.json", 'r') as sample_input_file:
        input_data5 = json.load(sample_input_file)
        result5 = parse_model_input_from_input_data_traditional(input_data5)
        assert '2' in result5
        assert result5['2'].shape == (2, 2, 3)

    with open("./resources/mlflow/unnamed_tensor_input.json", 'r') as sample_input_file:
        input_data6 = json.load(sample_input_file)
        result6 = parse_model_input_from_input_data_traditional(input_data6)
        assert result6.shape == (2, 2, 3)
        assert isinstance(result6, np.ndarray)


def test_parse_model_input_from_input_data_dict_to_pandas(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_traditional

    with open("./resources/mlflow/sample_2_0_input.txt", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = parse_model_input_from_input_data_traditional(input_data1)
        assert isinstance(result1, pd.DataFrame)
        assert len(result1) == 1
        assert result1.loc[0].at["a"] == 3.0
        assert result1.loc[0].at["c"] == "foo"


def test_parse_model_input_from_input_data_list_to_tensor(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_traditional

    input_data1 = [3.0, 42, "bar"]
    result1 = parse_model_input_from_input_data_traditional(input_data1)
    assert isinstance(result1, np.ndarray)
    assert result1.shape == (3,)


def test_parse_model_input_from_input_data_str(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_transformers

    with open("./resources/mlflow_unit/str_input_example.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = parse_model_input_from_input_data_transformers(input_data)
        assert isinstance(result, str)
        assert result == input_data["input_data"]


def test_parse_model_input_from_input_data_list_str(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_transformers

    with open("./resources/mlflow_unit/list_str_input_example.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = parse_model_input_from_input_data_transformers(input_data)
        assert isinstance(result, list)
        assert isinstance(result[0], str)
        assert result == input_data["input_data"]


def test_parse_model_input_from_input_data_list_dict_str(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_transformers

    with open("./resources/mlflow_unit/list_dict_str_input_example.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = parse_model_input_from_input_data_transformers(input_data)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert result == input_data["input_data"]


def test_parse_model_input_from_input_data_dict_str_list_str(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_transformers

    with open("./resources/mlflow_unit/dict_str_list_str_input_example.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = parse_model_input_from_input_data_transformers(input_data)
        assert isinstance(result, dict)
        assert isinstance(result["this"], str)
        assert isinstance(result["a"], list)
        assert isinstance(result["a"][0], str)
        assert result == input_data["input_data"]


def test_parse_model_input_from_input_data_list_dict_list_str(teardown):
    setup_traditional()
    from mlflow_score_script import parse_model_input_from_input_data_transformers

    with open("./resources/mlflow_unit/list_dict_list_str_input_example.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = parse_model_input_from_input_data_transformers(input_data)
        assert isinstance(result, list)
        assert isinstance(result[0], dict)
        assert result == input_data["input_data"]


def test_get_sample_input_from_loaded_example_pandas_2_0(teardown):
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


def test_get_sample_input_from_loaded_example_ndarray_named(teardown):
    setup_traditional()
    from mlflow_score_script import get_sample_input_from_loaded_example

    example_info = {
        "type": "ndarray",
    }

    with open("./resources/mlflow/mlflow_tensor_spec_named/input_example.json", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = get_sample_input_from_loaded_example(example_info, input_data1)
        assert isinstance(result1, dict)
        assert all(isinstance(value, np.ndarray) for value in result1.values())
        assert result1["1"].shape == (2, 2, 3)


def test_get_sample_input_from_loaded_example_ndarray_unnamed(teardown):
    setup_traditional()
    from mlflow_score_script import get_sample_input_from_loaded_example

    example_info = {
        "type": "ndarray",
    }

    with open("./resources/mlflow/mlflow_tensor_spec_unnamed/input_example.json", 'r') as sample_input_file:
        input_data1 = json.load(sample_input_file)
        result1 = get_sample_input_from_loaded_example(example_info, input_data1)
        assert isinstance(result1, np.ndarray)
        assert result1.shape == (2, 2, 3)


def test_get_sample_input_from_loaded_example_str_pandas(teardown):
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
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow/mlflow_model_folder/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        assert isinstance(result1, pd.DataFrame)
        assert result1.dtypes[0] == np.dtype('int64')
        assert result1.dtypes[1] == np.dtype('int64')


def test_get_samples_from_signature_named_tensor(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow/mlflow_tensor_spec_named/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        assert isinstance(result1, dict)
        assert isinstance(result1["2"], np.ndarray)
        assert result1["0"].shape == (1, 2, 3)


def test_get_samples_from_signature_unnamed_tensor(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow/mlflow_tensor_spec_unnamed/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        assert isinstance(result1, np.ndarray)
        assert result1.shape == (1, 2, 3)

@pytest.mark.parametrize("setup,result_type", [(setup_traditional, "dataframe"), (setup_transformers, "list")])
@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_get_samples_from_signature_str(teardown, setup, result_type):
    setup()
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow_unit/str_translation/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        if result_type == "dataframe":
            assert isinstance(result1, pd.DataFrame)
            assert result1.dtypes[0] == np.dtype('O')
        elif result_type == "list":
            assert isinstance(result1, list)
            assert len(result1) > 0
            assert isinstance(result1[0], str)
        else:
            assert False
        # TODO there's no actual example here, just an empty dataframe with object type


def test_get_samples_from_signature_dict_str_pandas(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature

    with open("./resources/mlflow_unit/dict_str_qa/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        result1, _, _ = get_samples_from_signature(input_sig)
        assert isinstance(result1, pd.DataFrame)
        assert result1.dtypes[0] == np.dtype('O')
        assert "question" in result1.columns and "context" in result1.columns
        # TODO also empty DataFrame, but columns have correct names and object type


def test_get_parameter_type_no_sample_no_sig(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature, get_parameter_type, NoSampleParameterType

    with open("./resources/mlflow/mlflow_model_folder_no_sig/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = None
        sample_input, _, _ = get_samples_from_signature(input_sig)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, NoSampleParameterType)
        assert result1.input_to_swagger() == {"type": "object", "example": {}}


def test_get_parameter_type_pandas(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature, get_sample_input_from_loaded_example, get_parameter_type

    with open("./resources/mlflow/mlflow_2_0_model_folder/input_example.json", 'r') as sample_input_file:
        example_info = {
            "type": "dataframe",
            "pandas_orient": "split"
        }
        input_data1 = json.load(sample_input_file)
        sample_input = get_sample_input_from_loaded_example(example_info, input_data1)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, PandasParameterType)
        assert result1.input_to_swagger() == {
            'type': 'object',
            'required': ['columns', 'index', 'data'],
            'properties': {
                'columns': {'type': 'array', 'items': {'type': 'string'}},
                'index': {'type': 'array', 'items': {'type': 'integer', 'format': 'int64'}},
                'data': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}}
            },
            'example': {'columns': ['a', 'b', 'c'], 'index': [0], 'data': [[3.0, 1, 'foo']]},
            'format': 'pandas.DataFrame:split'
        }

    with open("./resources/mlflow/mlflow_2_0_model_folder/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        sample_input, _, _ = get_samples_from_signature(input_sig)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, PandasParameterType)
        assert result1.input_to_swagger() == {
            'type': 'object',
            'required': ['columns', 'index', 'data'],
            # TODO note the differences here - can we do better?
            'properties': {
                'columns': {'type': 'array', 'items': {'type': 'string'}},
                'index': {'type': 'array', 'items': {'type': 'object'}},
                'data': {'type': 'array', 'items': {'type': 'object'}}
            },
            'example': {'columns': ['a', 'b', 'c'], 'index': [], 'data': []},
            'format': 'pandas.DataFrame:split'
        }


def test_get_parameter_type_unnamed_tensor(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature, get_sample_input_from_loaded_example, get_parameter_type

    with open("./resources/mlflow/mlflow_tensor_spec_unnamed/input_example.json", 'r') as sample_input_file:
        example_info = {
            "type": "ndarray",
        }
        input_data1 = json.load(sample_input_file)
        sample_input = get_sample_input_from_loaded_example(example_info, input_data1)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, NumpyParameterType)
        result1.input_to_swagger() == {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'number',
                            'format': 'double'
                        }
                    }
                }
            },
            'example': [[[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]]],
            'format': 'numpy.ndarray'
        }

    with open("./resources/mlflow/mlflow_tensor_spec_unnamed/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        sample_input, _, _ = get_samples_from_signature(input_sig)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, NumpyParameterType)
        result1.input_to_swagger() == {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'number',
                            'format': 'double'
                        }
                    }
                }
            },
            'example': [[[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]], [[[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]], [[0.0], [0.0], [0.0], [0.0], [0.0]]]],
            'format': 'numpy.ndarray'
        }


def test_get_parameter_type_named_tensor(teardown):
    setup_traditional()
    from mlflow_score_script import get_samples_from_signature, get_sample_input_from_loaded_example, get_parameter_type

    with open("./resources/mlflow/mlflow_tensor_spec_named/input_example.json", 'r') as sample_input_file:
        example_info = {
            "type": "ndarray",
        }
        input_data1 = json.load(sample_input_file)
        sample_input = get_sample_input_from_loaded_example(example_info, input_data1)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, StandardPythonParameterType)
        assert isinstance(result1.sample_input["0"], NumpyParameterType)
        swagger = result1.input_to_swagger()
        assert swagger["type"] == "object"
        assert swagger["required"] == ['0', '1', '2']
        assert swagger["properties"]["0"] == {
            'type': 'array',
            'items': {
                'items': {
                    'items': {
                        'format': 'double',
                        'type': 'number'
                    },
                    'type': 'array'
                },
                'type': 'array'
            },
            'format': 'numpy.ndarray'
        }
        assert "2" in swagger["example"]

    with open("./resources/mlflow/mlflow_tensor_spec_named/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        sample_input, _, _ = get_samples_from_signature(input_sig)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, StandardPythonParameterType)
        assert isinstance(result1.sample_input["0"], NumpyParameterType)
        swagger = result1.input_to_swagger()
        assert swagger["type"] == "object"
        assert swagger["required"] == ['0', '1', '2']
        assert swagger["properties"]["0"] == {
            'type': 'array',
            'items': {
                'items': {
                    'items': {
                        'format': 'double',
                        'type': 'number'
                    },
                    'type': 'array'
                },
                'type': 'array'
            },
            'format': 'numpy.ndarray'
        }
        assert "2" in swagger["example"]


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_get_parameter_type_str(teardown):
    setup_transformers()
    from mlflow_score_script import get_samples_from_signature, get_sample_input_from_loaded_example, get_parameter_type

    with open("./resources/mlflow_unit/str_translation/input_example.json", 'r') as sample_input_file:
        example_info = {
            "type": "dataframe",
            "pandas_orient": "split"
        }
        input_data1 = json.load(sample_input_file)
        sample_input = get_sample_input_from_loaded_example(example_info, input_data1)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, PandasParameterType)
        assert result1.input_to_swagger() == {
            'type': 'object',
            'required': ['columns', 'index', 'data'],
            'properties': {
                'columns': {'type': 'array', 'items': {'type': 'string'}},
                'index': {'type': 'array', 'items': {'type': 'integer', 'format': 'int64'}},
                'data': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}}}
            },
            'example': {'columns': ['data'], 'index': [0], 'data': [[['MLflow is great!']]]},
            'format': 'pandas.DataFrame:split'
        }

    with open("./resources/mlflow_unit/str_translation/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        sample_input, _, _ = get_samples_from_signature(input_sig)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, StandardPythonParameterType)
        assert result1.input_to_swagger() == {'type': 'array', 'items': {'type': 'string'}, 'example': ['sample string']}


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_get_parameter_type_dict_str_pandas(teardown):
    setup_transformers()
    from mlflow_score_script import get_samples_from_signature, get_sample_input_from_loaded_example, get_parameter_type

    with open("./resources/mlflow_unit/dict_str_qa/input_example.json", 'r') as sample_input_file:
        example_info = {
            "type": "dataframe",
            "pandas_orient": "split"
        }
        input_data1 = json.load(sample_input_file)
        sample_input = get_sample_input_from_loaded_example(example_info, input_data1)
        result1, _, _ = get_parameter_type(sample_input)
        assert isinstance(result1, PandasParameterType)
        assert result1.input_to_swagger() == {
            'type': 'object',
            'required': ['columns', 'index', 'data'],
            'properties': {
                'columns': {'type': 'array', 'items': {'type': 'string'}},
                'index': {'type': 'array', 'items': {'type': 'integer', 'format': 'int64'}},
                'data': {'type': 'array', 'items': {'type': 'array', 'items': {'type': 'string'}}}
            },
            'example': {
                'columns': ['question', 'context'],
                'index': [0],
                'data': [['Why is model conversion important?', 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.']]
            },
            'format': 'pandas.DataFrame:split'
        }

    with open("./resources/mlflow_unit/dict_str_qa/MLmodel", 'r') as mlmodel:
        loaded_dict = yaml.safe_load(mlmodel.read())
        input_sig = ModelSignature.from_dict(loaded_dict["signature"])
        sample_input, _, _ = get_samples_from_signature(input_sig)
        result1, params_param, _ = get_parameter_type(sample_input)
        assert isinstance(result1, StandardPythonParameterType)
        assert result1.input_to_swagger() == {
            'type': 'object',
            'required': ['question', 'context'],
            'properties': {'context': {'type': 'string'}, 'question': {'type': 'string'}},
            'example': {'context': 'sample string', 'question': 'sample string'}
        }
        assert params_param.input_to_swagger() == {'example': {}, 'type': 'object'}


@pytest.mark.parametrize("model_folder", ["distilbert-base-uncased-noex", "distilbert-base-uncased"])
@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_distilbert_with_and_without_input_example(model_folder):
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = model_folder
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/fill_mask_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_translation_t5_small():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "translation-t5-small"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/translation_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_question_answering_distilbert():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "question_answering_distilbert"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/qa_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_question_answering_multiple_distilbert():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "question_answering_distilbert"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/qa_input_multiple.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        try:
            result = mlflow_score_script.run(input_data["input_data"])
            assert isinstance(result, list)
            assert len(result) == 2
            assert isinstance(result[0], str)
        except ValueError as e:
            assert str(e) == "Invalid input data type to parse. Expected: <class 'dict'> but got <class 'list'>"


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_text_classification_deberta():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "text_classification_deberta"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/text_classification_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], dict)
        assert "label" in result[0]
        assert "score" in result[0]


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_summarization_distilbart():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "summarization_distilbart"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/summarization_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_token_classification_distilbert():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "token_classification_distilbert"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/token_classification_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_text_generation_gpt2():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "generation_gpt2"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    with open("./resources/mlflow_unit/text_generation_input.json", 'r') as sample_input_file:
        input_data = json.load(sample_input_file)
        result = mlflow_score_script.run(input_data["input_data"])
        assert isinstance(result, list)
        assert len(result) > 0
        assert isinstance(result[0], str)


def test_params():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "params"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    dataframe_dict = {
        "columns": [
            "sentence1"
        ],
        "data": [
            [ "Once upon a time " ]
        ],
        "index": [0]
    }
    dataframe = pd.read_json(
        json.dumps(dataframe_dict),
        # needs open source fix
        # orient=input_example_info['pandas_orient'],
        orient='split',
        dtype=False
    )
    params_dict = {
        "num_beams": 1,
        "max_length": 256
    }
    result = mlflow_score_script.run(dataframe, params_dict)
    assert result == "Once upon a time 256"
    _, params_param, _ = mlflow_score_script._get_schema_params()
    assert isinstance(params_param, StandardPythonParameterType)
    assert params_param.input_to_swagger() == {
        'example': {'max_length': 512, 'num_beams': 2},
        'properties': {
            'max_length': {'format': 'int64', 'type': 'integer'},
            'num_beams': {'format': 'int64', 'type': 'integer'}
        },
        'required': ['num_beams', 'max_length'],
        'type': 'object'
    }


def test_params_no_defaults():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "params_no_defaults"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    dataframe_dict = {
        "columns": [
            "sentence1"
        ],
        "data": [
            [ "Once upon a time " ]
        ],
        "index": [0]
    }
    dataframe = pd.read_json(
        json.dumps(dataframe_dict),
        # needs open source fix
        # orient=input_example_info['pandas_orient'],
        orient='split',
        dtype=False
    )
    params_dict = {
        "num_beams": 1,
        "max_length": 256
    }
    result = mlflow_score_script.run(dataframe, params_dict)
    assert result == "Once upon a time 256"
    _, params_param, _ = mlflow_score_script._get_schema_params()
    assert isinstance(params_param, StandardPythonParameterType)
    assert params_param.input_to_swagger() == {
        'example': {'max_length': 42, 'num_beams': 42},
        'properties': {
            'max_length': {'format': 'int64', 'type': 'integer'},
            'num_beams': {'format': 'int64', 'type': 'integer'}
        },
        'required': ['num_beams', 'max_length'],
        'type': 'object'
    }


def test_params_hftransformers_back_compat():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "params_hftransformers"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    input_dict = {
        "key1": ["beep"],
        "key2": ["boop"],
        "key3": ["-12"],
        "parameters": {
            "num_beams": 10,
            "max_length": 1024
        }
    }
    result = mlflow_score_script.run(input_dict)
    assert result == "boop -12 1024"
    _, params_param, _ = mlflow_score_script._get_schema_params()
    assert isinstance(params_param, StandardPythonParameterType)
    assert params_param.input_to_swagger() == {
        'example': {'max_length': 512, 'num_beams': 2},
        'properties': {
            'max_length': {'format': 'int64', 'type': 'integer'},
            'num_beams': {'format': 'int64', 'type': 'integer'}
        },
        'required': ['num_beams', 'max_length'],
        'type': 'object'
    }


@pytest.mark.skip(reason="Transformers model too large to add to git")
def test_params_hftransformersv2_back_compat():
    os.environ["AZUREML_MODEL_DIR"] = "./resources/mlflow_unit"
    os.environ["MLFLOW_MODEL_FOLDER"] = "params_hftransformersv2"
    try:
        del sys.modules['mlflow_score_script']
    except:
        pass
    try:
        del sys.modules['inference_schema']
    except:
        pass
    from inference_schema.schema_util import __functions_schema__
    __functions_schema__["mlflow_score_script.run"] = {}
    import mlflow_score_script
    mlflow_score_script.init()
    input_dict = {
        "input_string": ["I believe the meaning of life is"],
        "parameters": {
            "temperature": 0.6,
            "top_p": 0.9,
            "do_sample": True,
            "max_new_tokens": 96,
        }
    }
    result = mlflow_score_script.run(input_dict)
    assert result == ['Ich glaube, die Bedeutung des Lebens ist']
    _, params_param, _ = mlflow_score_script._get_schema_params()
    assert isinstance(params_param, StandardPythonParameterType)

    assert params_param.input_to_swagger() == {
        'type': 'object',
        'required': ['temperature', 'top_p', 'do_sample', 'max_new_tokens'],
        'properties': {
            'temperature': {'type': 'number', 'format': 'double'},
            'top_p': {'type': 'number', 'format': 'double'},
            'do_sample': {'type': 'boolean'},
            'max_new_tokens': {'type': 'integer', 'format': 'int64'}
        },
        'example': {'temperature': 0.6, 'top_p': 0.9, 'do_sample': True, 'max_new_tokens': 96}}
