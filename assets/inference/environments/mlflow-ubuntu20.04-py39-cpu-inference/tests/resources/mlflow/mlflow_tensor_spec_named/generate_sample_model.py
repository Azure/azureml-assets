import mlflow # using mlflow==1.24.0, also needs protobuf==3.20.3, numpy==1.22.2
# from mlflow.models import infer_signature
import numpy as np
import os

# define a custom model
class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input):
        return self.my_custom_function(model_input)

    def my_custom_function(self, model_input):
        # do something with the model input
        return np.array([0.1])

# save the model
my_model = MyModel()
# if you have an error about YAML, delete mlruns directory
with mlflow.start_run():
    from pathlib import Path
    import posixpath
    path = "./temporary"
    path = posixpath.normpath(path)
    input_arr = {
        "0": np.array([[[0, 1, .5], [1, 2, 3.5]], [[2, 3, 4.5], [3, 4, 5.5]]]),
        "1": np.array([[[0, 1, .5], [1, 2, 3.5]], [[2, 3, 4.5], [3, 4, 5.5]]]),
        "2": np.array([[[0, 1, .5], [1, 2, 3.5]], [[2, 3, 4.5], [3, 4, 5.5]]])
    }
    signature = mlflow.models.infer_signature(input_arr, my_model.predict(None, input_arr))
    mlflow.pyfunc.log_model(artifact_path=path, python_model=my_model, signature=signature, input_example=input_arr)