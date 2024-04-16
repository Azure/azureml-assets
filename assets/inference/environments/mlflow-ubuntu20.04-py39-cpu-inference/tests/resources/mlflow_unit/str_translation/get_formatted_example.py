# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from mlflow.models.signature import infer_signature
from mlflow.transformers import generate_signature_output
from transformers import pipeline
import mlflow

en_to_de = pipeline("translation_en_to_de")

data = "MLflow is great!"
output = generate_signature_output(en_to_de, data)
signature = infer_signature(data, output)

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model=en_to_de,
        artifact_path="english_to_german_translator",
        signature=signature,
        input_example=data,
    )