from mlflow.models.signature import infer_signature
from mlflow.transformers import generate_signature_output
from transformers import pipeline
import mlflow

model_name = "deepset/minilm-uncased-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

data = QA_input = {
    'question': 'Why is model conversion important?',
    'context': 'The option to convert models between FARM and transformers gives freedom to the user and let people easily switch between frameworks.'
}
model = nlp
output = generate_signature_output(model, data)
signature = infer_signature(data, output)

with mlflow.start_run() as run:
    mlflow.transformers.log_model(
        transformers_model=model,
        artifact_path="minilm_qa",
        signature=signature,
        input_example=data,
    )