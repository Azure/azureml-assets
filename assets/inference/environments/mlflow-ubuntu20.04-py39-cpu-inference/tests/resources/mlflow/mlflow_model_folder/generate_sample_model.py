# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""For generating the sample model."""
import mlflow   # using mlflow==1.24.0, also needs protobuf==3.20.3, numpy==1.22.2
# from mlflow.models import infer_signature
import pandas as pd


# define a custom model
class MyModel(mlflow.pyfunc.PythonModel):
    """For MyModel class."""
    def predict(self, context, model_input):
        """Predict."""
        return self.my_custom_function(model_input)

    def my_custom_function(self, model_input):
        """My custom function."""
        # do something with the model input
        return pd.DataFrame([0])


# save the model
my_model = MyModel()
# if you have an error about YAML, delete mlruns directory
with mlflow.start_run():
    import posixpath
    path = "./temporary"
    path = posixpath.normpath(path)
    input_df = pd.DataFrame([[0, 1]], columns=["a", "b"])
    signature = mlflow.models.infer_signature(input_df, my_model.predict(None, input_df))
    mlflow.pyfunc.log_model(artifact_path=path, python_model=my_model, signature=signature, input_example=input_df)
