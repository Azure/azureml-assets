# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import yaml

YAML_FILE = "spec.yaml"
BYPASS_GALLERY_CHECK = {
    "assets/promptflow/models/template-chat-flow/spec.yaml",
    "assets/promptflow/models/template-eval-flow/spec.yaml",
    "assets/promptflow/models/template-standard-flow/spec.yaml",
}

model_names = {}


def bypass_gallary_check(yaml_file):
    for file in BYPASS_GALLERY_CHECK:
        if os.path.samefile(yaml_file, file):
            return True


def check_spec_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        model_yaml = yaml.safe_load(file)

    properties = model_yaml.get("properties")
    if not properties.get("is-promptflow"):
        raise Exception(
            f"properties.is-promptflow must be True, it's {properties.get('is-promptflow')} in {yaml_file}")
    if properties.get("azureml.promptflow.section") != "gallery":
        if not bypass_gallary_check(yaml_file):
            raise Exception(
                f"properties.azureml.promptflow.section must be gallery, "
                f"it's {properties.get('azureml.promptflow.section')} in {yaml_file}")
    if properties.get("azureml.promptflow.type") not in {"chat", "evaluate", "standard"}:
        if not os.path.samefile(yaml_file, "assets/promptflow/models/bring-your-own-data-qna/spec.yaml"):
            raise Exception(
                f"properties.azureml.promptflow.type must in {'chat', 'evaluate', 'standard'}, "
                f"it's {properties.get('azureml.promptflow.type')} in {yaml_file}")
    name = properties.get("azureml.promptflow.name")
    if name in model_names.keys():
        raise Exception(
            f"Duplicated name found for fields properties.azureml.promptflow.name in {yaml_file}, "
            f"{name} already exists in {model_names.get(name)}.")
    else:
        model_names[name] = yaml_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models_dir", type=str)
    args = parser.parse_args()
    models_dir = args.models_dir
    examples = os.listdir(models_dir)
    errors = []
    print(f"Check model folders: {examples}")
    for example_dir in examples:
        try:
            check_spec_yaml(os.path.join(models_dir, example_dir, YAML_FILE))
        except Exception as e:
            errors.append(e)

    if len(errors) > 0:
        print(f"Found {len(errors)} errors when checking models' spec.yaml:")
        for error in errors:
            print(error)
        exit(1)
    else:
        print(
            f"Check passed. Found {len(errors)} errors when checking models' spec.yaml.")
        exit(0)
