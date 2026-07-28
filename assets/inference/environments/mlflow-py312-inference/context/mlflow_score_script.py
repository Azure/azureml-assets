"""Mlflow score script."""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import json
import logging
import numpy as np
import os
import mlflow
import inspect

from copy import deepcopy
from inference_schema.parameter_types.abstract_parameter_type import AbstractParameterType
from inference_schema.parameter_types.numpy_parameter_type import NumpyParameterType
from inference_schema.parameter_types.standard_py_parameter_type import StandardPythonParameterType
from inference_schema.schema_decorators import input_schema, output_schema
from mlflow.models import Model
from mlflow.pyfunc import load_model
from mlflow.pyfunc.scoring_server import _get_jsonable_obj
from azureml.ai.monitoring import Collector
from mlflow.types.utils import _infer_schema
from mlflow.types.schema import Schema, ColSpec, DataType
from mlflow.exceptions import MlflowException

_logger = logging.getLogger(__name__)

# Pandas installed, may not be necessary for tensorspec based models, so don't require it all the time
pandas_installed = False
try:
    import pandas as pd
    from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType

    pandas_installed = True
except ImportError:
    _logger.warning('Unable to import pandas')

param_schema_supported = False
try:
    from mlflow.types.schema import ParamSchema
    param_schema_supported = True
except ImportError:
    _logger.warning(f'Unable to import ParamSchema from MLflow. MLflow version: {mlflow.__version__}')


class NoSampleParameterType(AbstractParameterType):
    """NoSampleParameterType."""

    def __init__(self):
        """Init."""
        super(NoSampleParameterType, self).__init__(None)

    def deserialize_input(self, input_data):
        """Passthrough, do nothing to the incoming data."""
        return input_data

    def input_to_swagger(self):
        """Return schema for an empty object."""
        return {"type": "object", "example": {}}


def create_tensor_spec_sample_io(model_signature_io):
    """Create tensor spec sample."""
    _logger.info("Creating tensor spec sample")
    # Create a sample numpy.ndarray based on shape/type of the tensor info of the model
    io = model_signature_io.inputs
    if not model_signature_io.has_input_names():
        # If the input is not a named tensor, the sample io value that we create will just be a numpy.ndarray
        shape = io[0].shape
        if shape and shape[0] == -1:
            # -1 for first dimension means the input data is batched
            # Create a numpy array with the first dimension of shape as 1 so that inference-schema
            # can correctly generate the swagger sample for the input
            shape = list(deepcopy(shape))
            shape[0] = 1
        sample_io = np.zeros(tuple(shape), dtype=io[0].type)
    else:
        # otherwise, the input is a named tensor, so the sample io value that we create will be
        # Dict[str, numpy.ndarray], which maps input name to a numpy.ndarray of the corresponding size
        sample_io = {}
        for io_val in io:
            shape = io_val.shape
            if shape and shape[0] == -1:
                # -1 for first dimension means the input data is batched
                # Create a numpy array with the first dimension of shape as 1 so that inference-schema
                # can correctly generate the swagger sample for the input
                shape = list(deepcopy(shape))
                shape[0] = 1
            sample_io[io_val.name] = np.zeros(tuple(shape), dtype=io_val.type)
    return sample_io


def create_col_spec_sample_io(model_signature_io):
    """Create col spec sample."""
    _logger.info("Creating col spec sample")
    # Create a sample pandas.DataFrame based on shape/type of the tensor info of the model
    try:
        columns = model_signature_io.input_names()
    except AttributeError:  # MLflow < 1.24.0
        columns = model_signature_io.column_names()
    types = model_signature_io.pandas_types()
    schema = {}
    for c, t in zip(columns, types):
        schema[c] = t
    df = pd.DataFrame(columns=columns)
    return df.astype(dtype=schema)


def create_other_sample_io(model_signature_io):
    """Create other sample."""
    _logger.info("Creating 'other' (Python object) sample")
    inputs = model_signature_io.inputs
    sample_string = "sample string"
    if type(inputs[0]) is ColSpec and inputs[0].name is not None:
        # if isinstance(inputs, dict):
        _logger.info("Creating dict sample")
        sample_dict = {}
        for input in inputs:
            if input.type == DataType.boolean:
                sample_input = True
            elif input.type == DataType.string:
                sample_input = sample_string
            elif input.type == DataType.integer or input.type == DataType.long:
                sample_input = 42
            elif input.type == DataType.float or input.type == DataType.double:
                sample_input = 0.15
            else:
                _logger.info(f"Unhandled input type in dictionary value: {input.type}")
            sample_dict[input.name] = sample_input
        return sample_dict
    if isinstance(inputs, list):
        _logger.info("Creating list sample")
        sample_list = []
        if len(inputs) > 0 and (isinstance(inputs[0], str) or isinstance(inputs[0], ColSpec) and
                                inputs[0].type == DataType.string):
            sample_list.append(sample_string)
        return sample_list
    elif isinstance(inputs, str):
        _logger.info("Creating str sample")
        return sample_string
    raise "Unhandled data type when creating non colspec and non tensorspec sample"


def create_param_sample(model_signature_params):
    """Create param sample."""
    sample_params = {}
    if param_schema_supported and model_signature_params is not None and type(model_signature_params) is ParamSchema:
        for param in model_signature_params.params:
            if param.default is not None:
                sample_params[param.name] = param.default
            else:
                param_type = param.dtype
                if param_type == DataType.boolean:
                    sample_params[param.name] = True
                elif param_type == DataType.string:
                    sample_params[param.name] = "sample string"
                elif param_type == DataType.integer or param_type == DataType.long:
                    sample_params[param.name] = 42
                elif param_type == DataType.float or param_type == DataType.double:
                    sample_params[param.name] = 0.15
                else:
                    _logger.info(f"Unhandled param type: {param_type}")
    return sample_params


model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), os.getenv("MLFLOW_MODEL_FOLDER"))

# model loaded here using mlfow.models import Model so we have access to the model signature
model = Model.load(model_path)

is_hfv2 = "hftransformersv2" in model.flavors
_logger.info(f"is_hfv2: {is_hfv2}")
is_transformers = "transformers" in model.flavors
_logger.info(f"is_transformers: {is_transformers}")
is_langchain = "langchain" in model.flavors
_logger.info(f"is_langchain: {is_langchain}")
is_openai = "openai" in model.flavors
_logger.info(f"is_openai: {is_openai}")

try:
    metadata = model.metadata
except Exception as e:
    _logger.warning(f"failed to fetch metadata with err: {str(e)}")
    metadata = None

if metadata and metadata.get('is_acft_model', False):
    base_model_name = metadata.get("base_model_name", None)
    if base_model_name:
        _logger.info(f"base_model_name: {base_model_name}")
    base_model_task = metadata.get("base_model_task", None)
    if base_model_task:
        _logger.info(f"base_model_task: {base_model_task}")
    base_model_asset_id = metadata.get("base_model_asset_id", None)
    if base_model_asset_id:
        _logger.info(f"base_model_asset_id: {base_model_asset_id}")
    finetuning_task = metadata.get("finetuning_task", None)
    if finetuning_task:
        _logger.info(f"finetuning_task: {finetuning_task}")
    is_finetuned = metadata.get("is_finetuned_model", None)
    if is_finetuned is not None:
        _logger.info(f"is_finetuned_model: {is_finetuned}")

sample_input = None
input_param = None
sample_output = None
output_param = None
sample_params = None
params_param = None


def get_sample_input_from_loaded_example(input_example_info, loaded_input):
    """Get sample input from loaded example."""
    orient = "split" if "columns" in loaded_input else "values"
    if input_example_info['type'] == 'dataframe':
        _logger.info("Getting sample from loaded dataframe example")
        sample_input = pd.read_json(
            json.dumps(loaded_input),
            # needs open source fix
            # orient=input_example_info['pandas_orient'],
            orient=orient,
            dtype=False
        )
    elif input_example_info["type"] == "ndarray":
        _logger.info("Getting sample from loaded numpy example")
        inputs = loaded_input["inputs"]
        if isinstance(inputs, dict):
            sample_input = {
                input_name: np.asarray(input_value) for input_name, input_value in inputs.items()
            }
        else:
            sample_input = np.asarray(inputs)
    else:
        _logger.info("Getting sample from loaded python object example")
        # currently unused, as type always comes through from MLflow _Example creation as ndarray or dataframe
        sample_input = loaded_input
        _logger.warning('Potentially unable to handle sample model input of type "{}". The type must be one '
                        'of the list detailed in the MLflow repository: '
                        'https://github.com/mlflow/mlflow/blob/master/mlflow/types/utils.py#L91 '
                        '"dataframe" or "ndarray" is guaranteed to work best. For more information, please see: '
                        'https://aka.ms/aml-mlflow-deploy."'.format(model.saved_input_example_info['type']))
    return sample_input


# If a sample input is provided, load this input and use this as the sample input to create the
# scoring script and inference-schema decorators instead of creating a sample based on just the
# signature information
try:
    if model.saved_input_example_info:
        sample_input_file_path = os.path.join(model_path, model.saved_input_example_info['artifact_path'])
        with open(sample_input_file_path, 'r') as sample_input_file:
            _logger.info(f"loading example from file path: {sample_input_file_path}")
            loaded_input = json.load(sample_input_file)
            sample_input = get_sample_input_from_loaded_example(model.saved_input_example_info, loaded_input)
except Exception as e:
    _logger.warning(
        "Failure processing model sample input: {}.\nWill attempt to create sample input based on model signature. "
        "For more information, please see: https://aka.ms/aml-mlflow-deploy.".format(e)
    )


def get_samples_from_signature(
        model_signature_x,
        previous_sample_input=None,
        previous_sample_output=None,
        previous_sample_params=None):
    """Get samples from signature."""
    if model_signature_x is None:
        _logger.info("No model signature, returning previous sample input and output")
        return previous_sample_input, previous_sample_output, previous_sample_params
    model_signature_inputs = model_signature_x.inputs
    model_signature_outputs = model_signature_x.outputs
    try:
        model_signature_params = model_signature_x.params
    except AttributeError:
        _logger.info(f"Params not available on model signature. Setting to None. MLflow version: {mlflow.__version__}")
        model_signature_params = None
    if model_signature_inputs == Schema([ColSpec("string")]) and is_transformers:
        _logger.info("Getting sample from a transformers model taking string input")
        sample_input_x = create_other_sample_io(model_signature_inputs)
    elif type(model_signature_inputs.inputs[0]) is ColSpec and \
            model_signature_inputs.inputs[0].name is not None and is_transformers:
        _logger.info("Getting sample from a transformers model taking dict input")
        sample_input_x = create_other_sample_io(model_signature_inputs)
    elif model_signature_inputs and previous_sample_input is None:
        _logger.info("Getting sample from non-transformers model")
        if model_signature_inputs.is_tensor_spec():
            sample_input_x = create_tensor_spec_sample_io(model_signature_inputs)
        else:
            try:
                sample_input_x = create_col_spec_sample_io(model_signature_inputs)
            except:  # noqa: E722
                sample_input_x = create_other_sample_io(model_signature_inputs)
                _logger.warning("Sample input could not be parsed as either TensorSpec"
                                " or ColSpec. Falling back to taking the sample as is rather than"
                                " converting to numpy arrays or DataFrame.")
    else:
        _logger.info("Using previous sample input")
        sample_input_x = previous_sample_input

    if model_signature_outputs and previous_sample_output is None:
        if model_signature_outputs.is_tensor_spec():
            sample_output_x = create_tensor_spec_sample_io(model_signature_outputs)
        else:
            sample_output_x = create_col_spec_sample_io(model_signature_outputs)
    else:
        sample_output_x = previous_sample_output

    if model_signature_params and previous_sample_params is None:
        sample_params_x = create_param_sample(model_signature_params)
    else:
        sample_params_x = previous_sample_params

    return sample_input_x, sample_output_x, sample_params_x


# Handle the signature information to attempt creation of a sample based on signature if no concrete
# sample was provided
model_signature = model.signature
if model_signature:
    sample_input, sample_output, sample_params = get_samples_from_signature(
        model_signature, sample_input, sample_output, sample_params)
else:
    _logger.warning(
        "No signature information provided for model. If no sample information was provided with the model "
        "the deployment's swagger will not include input and output schema and typing information."
        "For more information, please see: https://aka.ms/aml-mlflow-deploy."
    )


def get_parameter_type(sample_input_ex, sample_output_ex=None, sample_param_ex=None):
    """Get parameter type."""
    if sample_input_ex is None:
        _logger.info("sample input is none, returning NoSampleParameterType")
        input_param = NoSampleParameterType()
    else:
        try:
            # schema = _infer_schema(sample_input_ex)
            # schema_types = schema.input_types
            _infer_schema(sample_input_ex)
        except MlflowException:
            pass
        finally:
            if isinstance(sample_input_ex, np.ndarray):
                _logger.info("sample input is a numpy array")
                # Unnamed tensor input
                input_param = NumpyParameterType(sample_input_ex, enforce_shape=False)
            elif pandas_installed and isinstance(sample_input_ex, pd.DataFrame):
                _logger.info("sample input is a dataframe")
                # TODO check with OSS about pd.Series
                input_param = PandasParameterType(sample_input_ex, enforce_shape=False, orient='split')
            # elif schema_types and isinstance(sample_input_ex, dict) and \
            #     not all(stype == DataType.string for stype in schema_types) and \
            #     all(isinstance(value, list) for value in sample_input_ex.values()):
            #     # for dictionaries where there is any non-string type, named tensor
            #     param_arg = {}
            #     for key, value in sample_input_ex.items():
            #         param_arg[key] = NumpyParameterType(value, enforce_shape=False)
            #     input_param = StandardPythonParameterType(param_arg)
            elif isinstance(sample_input_ex, dict) and is_transformers:
                input_param = StandardPythonParameterType(sample_input_ex)
            elif isinstance(sample_input_ex, dict):
                _logger.info("sample input is a dict")
                # TODO keeping this around while _infer_schema doesn't work on dataframe string signatures
                param_arg = {}
                for key, value in sample_input_ex.items():
                    param_arg[key] = NumpyParameterType(value, enforce_shape=False)
                input_param = StandardPythonParameterType(param_arg)
            elif isinstance(sample_input_ex, list) and is_transformers:
                _logger.info("transformers sample input is a list")
                input_param = StandardPythonParameterType(sample_input_ex)
            else:
                _logger.info("sample input is string, bytes, or non-transformers list")
                # strings, bytes, lists and dictionaries with only strings as base type
                input_param = NoSampleParameterType()

    if sample_output_ex is None:
        output_param = NoSampleParameterType()
    else:
        if isinstance(sample_output_ex, np.ndarray):
            # Unnamed tensor input
            output_param = NumpyParameterType(sample_output_ex, enforce_shape=False)
        elif isinstance(sample_output_ex, dict):
            param_arg = {}
            for key, value in sample_output_ex.items():
                param_arg[key] = NumpyParameterType(value, enforce_shape=False)
            output_param = StandardPythonParameterType(param_arg)
        else:
            output_param = PandasParameterType(sample_output_ex, enforce_shape=False, orient='records')

    if sample_param_ex is None:
        param_param = NoSampleParameterType()
    else:
        param_param = StandardPythonParameterType(sample_param_ex)

    return input_param, output_param, param_param


input_param, output_param, params_param = get_parameter_type(sample_input, sample_output, sample_params)

_logger.info(f"loading model from model path: {model_path}")
# we use mlflow.pyfunc's load_model function because it has a predict function on it we need for inferencing
model = load_model(model_path)


def init():
    """Init."""
    _logger.info("Initializing MLflow scoring script")
    global inputs_collector, outputs_collector
    try:
        inputs_collector = Collector(name='model_inputs')
        outputs_collector = Collector(name='model_outputs')
        _logger.info("Input and output collector initialized")
    except Exception as e:
        _logger.error("Error initializing model_inputs collector and model_outputs collector. {}".format(e))


@input_schema("input_data", input_param)
@input_schema("params", params_param, optional=True)
@output_schema(output_param)
def run(input_data, params=None):
    """Run."""
    _logger.info("Entering run function in MLflow scoring script")
    context = None

    # to support customers transitioning from hftransformersv2
    if params is None and "parameters" in input_data and is_transformers:
        params = input_data["parameters"]
        del input_data["parameters"]

        remaining_keys = list(input_data.keys())
        if len(remaining_keys) == 1 and remaining_keys[0] == "input_string":
            input_data = input_data["input_string"]

    if (
        isinstance(input_data, np.ndarray)
        or (isinstance(input_data, dict) and input_data and isinstance(list(input_data.values())[0], np.ndarray))
        or (pandas_installed and isinstance(input_data, pd.DataFrame))
    ):
        _logger.info("Predicting for dataframe and ndarray")
        # Collect model input
        try:
            context = inputs_collector.collect(input_data)
        except Exception as e:
            _logger.error("Error collecting model_inputs collection request. {}".format(e))

        if inspect.signature(model.predict).parameters.get("params"):
            result = model.predict(input_data, params=params)
        else:
            _logger.warning("Switching back to use a model.predict() without params. " +
                            "Likely an older version of MLflow in use. MLflow version: {mlflow.__version__}")
            result = model.predict(input_data)

        # Collect model output
        try:
            mdc_output_df = pd.DataFrame(result)
            outputs_collector.collect(mdc_output_df, context)
        except Exception as e:
            _logger.error("Error collecting model_outputs collection request. {}".format(e))

        return _get_jsonable_obj(result, pandas_orient="records")

    if is_transformers or is_langchain or is_openai:
        _logger.info("Parsing model input for LLMs")
        input = parse_model_input_from_input_data_transformers(input_data)
    else:
        _logger.info("Parsing model input for traditional models")
        input = parse_model_input_from_input_data_traditional(input_data)

    # Collect model input
    try:
        context = inputs_collector.collect(input)
    except Exception as e:
        _logger.error("Error collecting model_inputs collection request. {}".format(e))

    if inspect.signature(model.predict).parameters.get("params"):
        result = model.predict(input, params=params)
    else:
        _logger.warning("Switching back to use a model.predict() without params. " +
                        "Likely an older version of MLflow in use. MLflow version: {mlflow.__version__}")
        result = model.predict(input)

    # Collect output data
    try:
        mdc_output_df = pd.DataFrame(result)
        outputs_collector.collect(mdc_output_df, context)
    except Exception as e:
        _logger.error("Error collecting model_outputs collection request. {}".format(e))

    return _get_jsonable_obj(result, pandas_orient="records")


def parse_model_input_from_input_data_traditional(input_data):
    """Parse model input from input data traditional."""
    # Format input
    if isinstance(input_data, str):
        input_data = json.loads(input_data)
    if 'input_data' in input_data:
        input_data = input_data['input_data']
    if is_hfv2:
        input = input_data
    elif isinstance(input_data, list):
        # if a list, assume the input is a numpy array
        input = np.asarray(input_data)
    elif isinstance(input_data, dict) and "columns" in input_data and "index" in input_data and "data" in input_data:
        # if the dictionary follows pandas split column format, deserialize into a pandas Dataframe
        input = pd.read_json(json.dumps(input_data), orient="split", dtype=False)
    else:
        # otherwise, assume input is a named tensor, and deserialize into a dict[str, numpy.ndarray]
        input = {input_name: np.asarray(input_value) for input_name, input_value in input_data.items()}
    return input


def parse_model_input_from_input_data_transformers(input_data):
    """Parse model input from input data transformers."""
    # Format input
    if isinstance(input_data, str):
        _logger.info("input data is str")
        try:
            input_data = json.loads(input_data)
        except ValueError:
            # allow non-json strings to go through
            input = input_data

    if isinstance(input_data, dict) and 'input_data' in input_data:
        _logger.info("getting inputs out of input dictionary")
        input_data = input_data['input_data']

    if is_hfv2:
        _logger.info("input passed through directly for hfv2 models")
        input = input_data
    elif isinstance(input_data, str) or isinstance(input_data, bytes):
        _logger.info("input passed through directly for str/bytes")
        # strings and bytes go through
        input = input_data
    elif isinstance(input_data, list) and all(isinstance(element, str) for element in input_data):
        _logger.info("input passed through directly for lists of strings")
        # lists of strings go through
        input = input_data
    elif (
        isinstance(input_data, list)
        and all(isinstance(element, dict) for element in input_data)
    ):
        _logger.info("input passed through directly for dictionaries of strings")
        # lists of dicts of [str: str | List[str]] go through
        try:
            for dict_input in input_data:
                _validate_input_dictionary_contains_only_strings_and_lists_of_strings(dict_input)
            input = input_data
        except MlflowException:
            _logger.error("Could not parse model input - passed a list of dictionaries" +
                          " which had entries which were not strings or lists.")
    elif isinstance(input_data, list):
        _logger.info("assuming list is a numpy array")
        # if a list, assume the input is a numpy array
        input = np.asarray(input_data)
    elif isinstance(input_data, dict) and "columns" in input_data and "index" in input_data and "data" in input_data:
        _logger.info("coercing input to dataframe")
        # if the dictionary follows pandas split column format, deserialize into a pandas Dataframe
        input = pd.read_json(json.dumps(input_data), orient="split", dtype=False)
    elif isinstance(input_data, dict):
        # if input is a dictionary, but is not all ndarrays and is not pandas, it must only contain strings
        try:
            _validate_input_dictionary_contains_only_strings_and_lists_of_strings(input_data)
            _logger.info("input passed through directly for dicts/lists of strings")
            input = input_data
        except MlflowException:
            _logger.info("deserializing input as a named tensor")
            # otherwise, assume input is a named tensor, and deserialize into a dict[str, numpy.ndarray]
            input = {input_name: np.asarray(input_value) for input_name, input_value in input_data.items()}
    else:
        _logger.info(f"Input did not match any particular format. Input type: {type(input_data)}")
        input = input_data

    return input


# vendored from MLflow OSS
def _validate_input_dictionary_contains_only_strings_and_lists_of_strings(data):
    invalid_keys = []
    invalid_values = []
    value_type = None
    for key, value in data.items():
        if not value_type:
            value_type = type(value)
        if isinstance(key, bool):
            invalid_keys.append(key)
        elif not isinstance(key, (str, int)):
            invalid_keys.append(key)
        if isinstance(value, list) and not all(isinstance(item, (str, bytes)) for item in value):
            invalid_values.append(key)
        elif not isinstance(value, (np.ndarray, list, str, bytes)):
            invalid_values.append(key)
        elif isinstance(value, np.ndarray) or value_type == np.ndarray:
            if not isinstance(value, value_type):
                invalid_values.append(key)
    if invalid_values:
        from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
        raise MlflowException(
            "Invalid values in dictionary. If passing a dictionary containing strings, all "
            "values must be either strings or lists of strings. If passing a dictionary containing "
            "numeric values, the data must be enclosed in a numpy.ndarray. The following keys "
            f"in the input dictionary are invalid: {invalid_values}",
            error_code=INVALID_PARAMETER_VALUE,
        )
    if invalid_keys:
        raise MlflowException(
            f"The dictionary keys are not all strings or indexes. Invalid keys: {invalid_keys}"
        )


# for testing purposes, return input, parameter, and output params
def _get_schema_params():
    return input_param, params_param, output_param
