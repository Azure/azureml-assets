# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

"""
Publish to registry
"""

import yaml
import argparse
from typing import Dict

from azure.core.exceptions import ResourceExistsError
from azure.ai.ml import MLClient
from azure.ai.ml.entities import Environment
from azure.identity import DefaultAzureCredential, ClientSecretCredential, InteractiveBrowserCredential
from azure.ai.ml.entities._load_functions import (
    load_environment,
    load_component,
)


# Replace hard coding components with regex
COMMAND_COMPONENTS = [
    # Model Selector
    {
        "component_path": "finetune_acft_hf_nlp/components/model_import/text_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/model_import/token_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/model_import/translation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/model_import/summarization/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/model_import/question_answering/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/model_import/text_generation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    # {
    #     "component_path": "finetune_acft_hf_diffusion/components/model_import/stable_diffusion/spec.yaml",
    #     "environment_path": "finetune_acft_hf_diffusion/environments/acpt/spec.yaml"
    # },

    # Preprocess
    {
        "component_path": "finetune_acft_hf_nlp/components/preprocess/text_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/preprocess/token_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/preprocess/summarization/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/preprocess/translation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/preprocess/question_answering/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/preprocess/text_generation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    # {
    #     "component_path": "finetune_acft_hf_diffusion/components/preprocess/stable_diffusion/spec.yaml",
    #     "environment_path": "finetune_acft_hf_diffusion/environments/acpt/spec.yaml"
    # },

    # Finetune
    {
        "component_path": "finetune_acft_hf_nlp/components/finetune/text_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/finetune/token_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/finetune/summarization/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/finetune/translation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/finetune/question_answering/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    # {
    #     "component_path": "finetune_acft_hf_diffusion/components/finetune/stable_diffusion/spec.yaml",
    #     "environment_path": "finetune_acft_hf_diffusion/environments/acpt/spec.yaml"
    # }
    {
        "component_path": "finetune_acft_hf_nlp/components/validation/common/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
     {
        "component_path": "finetune_acft_hf_nlp/components/model_converter/common/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/finetune/text_generation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
]


PIPELINE_COMPONENTS = [
    # Pipeline components
    {
        "component_path": "finetune_acft_hf_nlp/components/pipeline_components/question_answering/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/pipeline_components/summarization/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/pipeline_components/text_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/pipeline_components/token_classification/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/pipeline_components/translation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    },
    {
        "component_path": "finetune_acft_hf_nlp/components/pipeline_components/text_generation/spec.yaml",
        "environment_path": "finetune_acft_hf_nlp/environments/acpt/spec.yaml"
    }
]


def register_environments(ml_client: MLClient, version) -> Dict[str, Environment]:
    """
    Create environment with given requirements
    """
    all_env_paths = set()
    for metadata in COMMAND_COMPONENTS:
        all_env_paths.add(metadata["environment_path"])

    registered_envs = {}
    for env_path in all_env_paths:
        print(env_path)
        tmp_env_path = env_path.replace('.yaml', '_tmp.yaml')
        with open(env_path, 'r') as rptr:
            env_data = yaml.safe_load(rptr)

        # update name and version for environment
        env_data["version"] = version
        env_data["name"] = "acft-hf-nlp-gpu"

        # setting the build context for docker build
        del env_data["build"]
        env_data["build"] = {}
        env_data["build"]["path"] = "context"

        # save the update yaml in a temp location
        with open(tmp_env_path, 'w') as wptr:
            yaml.dump(env_data, wptr, default_flow_style=False)

        env: Environment = load_environment(tmp_env_path)
        print(env)
        try:
            reg_env = ml_client.environments.create_or_update(env)
            registered_envs[env_path] = reg_env
        except ResourceExistsError as e:
            print("WARNING!", e)
            registered_envs[env_path] = env

    return registered_envs


def register_command_components(ml_client: MLClient, registered_envs, args_comp_version):
    """
    Register components in workspace
    """
    component_name_version_map = {}
    for metadata in COMMAND_COMPONENTS:
        component_path = metadata["component_path"]
        environment_path = metadata["environment_path"]

        print("Loading component: ", component_path)
        comp = load_component(component_path)

        if registered_envs is not None:
            reg_env = registered_envs[environment_path]
            ver = reg_env.version
            major, minor, patch = ver.split(".")
            patch_version = int(patch)
            comp.version = "{}.{}.{}".format(major, minor, patch_version)
            comp.environment = reg_env
        else:
            ver = args_comp_version
            major, minor, patch = ver.split(".")
            patch_version = int(patch)
            comp.version = "{}.{}.{}".format(major, minor, patch_version)

        try:
            reg_component = ml_client.components.create_or_update(comp)
            print(f"Component name: {reg_component.name}, version: {reg_component.version}")
        except ResourceExistsError as e:
            print("WARNING!", e)

        component_name_version_map[reg_component.name] = reg_component.version

    return component_name_version_map, comp.version


def register_pipeline_components(ml_client: MLClient, component_name_version_map, comp_version):
    """
    Register components in workspace
    """

    for metadata in PIPELINE_COMPONENTS:
        component_path = metadata["component_path"]
        tmp_component_path = component_path.replace('.yaml', '_tmp.yaml')
        with open(component_path, 'r') as rptr:
            component_data = yaml.safe_load(rptr)
        # make changes to pipeline component yaml
        component_data["version"] = comp_version
        for job_name in component_data["jobs"]:
            if job_name in component_name_version_map:
                job_version = component_name_version_map[job_name]
                component_data["jobs"][job_name]["component"] = f"azureml:{job_name}:{job_version}"
        # save the update yaml in a temp location
        with open(tmp_component_path, 'w') as wptr:
            yaml.dump(component_data, wptr, default_flow_style=False, sort_keys=False)
        print("Loading component: ", tmp_component_path)
        pipeline_comp = load_component(tmp_component_path)
        pipeline_comp.version = comp_version

        try:
            reg_component = ml_client.components.create_or_update(pipeline_comp)
            print(f"Component name: {reg_component.name}, version: {reg_component.version}")
        except ResourceExistsError as e:
            print("WARNING!", e)


def main(args):

    try:
        credential = ClientSecretCredential(args.sp_tenant_id, args.sp_client_id, args.sp_client_secret)
    except Exception as _:
        credential = DefaultAzureCredential()
        #credential = InteractiveBrowserCredential()

    ml_client = MLClient(
        credential,
        subscription_id=args.subscription_id,
        resource_group_name=args.resource_group_name,
        workspace_name=args.workspace_name
        #registry_name="azureml-preview-test1"
    )
    print(ml_client)

    # register the environment
    if args.publish_env.lower() == "true":
        print("Registering environment")
        registered_envs = register_environments(ml_client, args.version)
    else:
        print("skipping environment publishing")
        registered_envs = None

    # register command components
    print("Registering command components")
    component_name_version_map, comp_version = register_command_components(ml_client, registered_envs, args.version)

    # register pipeline components
    print("Registering pipeline components")
    register_pipeline_components(ml_client, component_name_version_map, comp_version)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sp_tenant_id",
        dest="sp_tenant_id",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--sp_client_id",
        dest="sp_client_id",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--sp_client_secret",
        dest="sp_client_secret",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--subscription_id",
        dest="subscription_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--resource_group_name",
        dest="resource_group_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--workspace_name",
        dest="workspace_name",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--registry_name",
        dest="registry_name",
        type=str,
        required=False,
    )

    parser.add_argument(
        "--publish_env",
        type=str,
        default="true",
        required=False,
    )

    parser.add_argument(
        "--version",
        type=str,
        default=None,
        required=True,
    )

    args = parser.parse_args()
    print(args)

    main(args)
