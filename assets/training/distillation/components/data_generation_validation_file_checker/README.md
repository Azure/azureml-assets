# OSS Distillation Data Generation Batch Scoring Selector

This component is designed to check if the validation file is present or not in the OSS Distillation pipeline. It ensures that the provided validation file path is valid and accessible.

## Description

The OSS Distillation Data Generation Batch Scoring Selector is a command component that verifies the presence of a validation file. It supports various data formats including `jsonl`, `json`, `csv`, `tsv`, and `parquet`.

## Environment

- **Environment**: `azureml://registries/azureml/environments/model-evaluation/labels/latest`

## Inputs

| Name                  | Type     | Optional | Description                                                                                                 |
|-----------------------|----------|----------|-------------------------------------------------------------------------------------------------------------|
| validation_file_path  | uri_file | Yes      | Path to the registered validation data asset. The supported data formats are `jsonl`, `json`, `csv`, `tsv`, and `parquet`. |

## Outputs

| Name   | Type    | Description                        |
|--------|---------|------------------------------------|
| output | boolean | Indicates if the validation file is present or not. |