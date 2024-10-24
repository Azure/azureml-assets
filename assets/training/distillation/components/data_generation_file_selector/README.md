# OSS Distillation Generate Data Path Selection

## Description
This component selects the path for data generation based on the task type. It supports various data generation tasks such as Natural Language Inference (NLI), Conversation, Natural Language Understanding for Question Answering (NLU_QA), Math, and Summarization.

## Environment
The component uses the following environment:
- `azureml://registries/azureml/environments/model-evaluation/labels/latest`

## Inputs
The component accepts the following inputs:

- `data_generation_task_type` (string): Specifies the type of data generation task. Supported values are:
  - `NLI`: Generate Natural Language Inference data
  - `CONVERSATION`: Generate conversational data (multi/single turn)
  - `NLU_QA`: Generate Natural Language Understanding data for Question Answering data
  - `MATH`: Generate Math data for numerical responses
  - `SUMMARIZATION`: Generate Key Summary for an Article

## Outputs
The component produces the following output:

- `output` (boolean): A control output indicating the success or failure of the data path selection.
