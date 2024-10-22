# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import mlflow
import os


# define a custom model
class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input, params):
        return self.my_custom_function(model_input, params)

    def my_custom_function(self, model_input, parameters):
        # do something with the model input
        return f"{model_input['key2'][0][0]} {model_input['key3'][0][0]} {parameters['max_length']}"


# save the model
my_model = MyModel()
# if you have an error about YAML, delete mlruns directory
# shutil.rmtree("mlruns")
# os.rmdir("mlruns")
with mlflow.start_run():
    from pathlib import Path
    import posixpath
    path = os.path.join(Path.cwd(), ".")
    path = posixpath.normpath(path)
    input_dict = {
        "key1": ["sentence1"],
        "key2": ["this is a string starting with"],
        "key3": ["0"]
    }
    params_dict = {
        "num_beams": 2,
        "max_length": 512
    }

    output = my_model.predict(None, input_dict, params_dict)
    signature = mlflow.models.infer_signature(input_dict, output, params_dict)

    mlflow.pyfunc.log_model(artifact_path=path, python_model=my_model, input_example=input_dict, signature=signature)
