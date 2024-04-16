# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# using mlflow==1.29.0, mlflow-skinny==1.29.0, azureml-mlflow==1.45.0
from mlflow.models.signature import infer_signature
from sklearn import datasets, linear_model
import mlflow
from azureml.opendatasets import Diabetes

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True, as_frame=True)
regr = linear_model.LinearRegression()
regr.fit(diabetes_X, diabetes_y)

model_name = "sklearn_no_params"

data = diabetes_X
model = regr
output = model.predict(data)
signature = infer_signature(data, output)

with mlflow.start_run() as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="no_params",
        signature=signature,
        input_example=data,
    )