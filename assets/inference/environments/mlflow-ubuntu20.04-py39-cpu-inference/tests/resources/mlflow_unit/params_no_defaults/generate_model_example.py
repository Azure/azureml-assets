# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""For generating model example."""
import mlflow
import os
import json
import pandas as pd


# define a custom model
class MyModel(mlflow.pyfunc.PythonModel):
    """For MyModel class."""
    
    def predict(self, context, model_input, params):
        """Predict."""
        return self.my_custom_function(model_input, params)

    def my_custom_function(self, model_input, parameters):
        """My custom function."""
        # do something with the model input
        return str(model_input['sentence1'].values[0]) + str(parameters["max_length"])


# save the model
my_model = MyModel()
# if you have an error about YAML, delete mlruns directory
# shutil.rmtree("mlruns")
# os.rmdir("mlruns")
with mlflow.start_run():
    from pathlib import Path
    import posixpath
    path = os.path.join(Path.cwd(), "../resources/mlflow_unit/params_dummy/")
    path = posixpath.normpath(path)
    dataframe_dict = {
        "columns": [
            "sentence1"
        ],
        "data": [
            ["this is a string starting with"]
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
            "num_beams": 2,
            "max_length": 512
        }

    output = my_model.predict(None, dataframe, params_dict)
    signature = mlflow.models.infer_signature(dataframe, output, params_dict)
    # take additional step here to remove param defaults from the MLmodel file

    mlflow.pyfunc.log_model(artifact_path=path, python_model=my_model, input_example=dataframe, signature=signature)
