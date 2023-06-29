# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for the Azure Cognitive Search Data Import."""
import os
from dataclasses import dataclass
from azure.identity import DefaultAzureCredential
from azure.search.documents import SearchClient
from langchain.docstore.document import Document
import argparse
import json
import yaml
from pathlib import Path
from azureml.rag.utils.azureml import get_workspace_from_environment
from azureml.rag.embeddings import EmbeddingsContainer
from azureml.rag.utils.connections import get_connection_by_id_v2, get_connection_credential


@dataclass
class ACSConnection:
    """ACSConnection Class."""

    credential: object
    endpoint: str
    index_name: str
    configuration_name: str
    content_key: str
    source_key: str
    title_key: str
    api_version: str


class ACS:
    """Class for interacting with Azure Cognitive Search."""

    def __init__(self, config: ACSConnection):
        """Initialize the class."""
        self.config = config
        self.configuration_name = config.configuration_name
        self.client = SearchClient(
            endpoint=config.endpoint,
            index_name=config.index_name,
            credential=config.credential,
            api_version=config.api_version
        )

    @staticmethod
    def from_config(config: ACSConnection):
        """from_config."""
        return ACS(config=config)

    def search(self,
               query,
               query_type="semantic",
               query_language="en-us",
               num_docs=1):
        """search."""
        print(f"Getting top {num_docs} docs from Azure Cognitive Search, query {query}")
        results = self.client.search(
            search_text=query,
            query_type=query_type,
            query_language=query_language,
            semantic_configuration_name=self.configuration_name,
            top=num_docs)
        docs_score = []
        for idx, result in enumerate(results):
            print(result)
            metadata = {
                "source": result.get(self.config.source_key, f"Source {idx}"),
                "score": result["@search.score"]}
            docs_score.append((
                Document(page_content=result[self.config.content_key], metadata=metadata),
                result["@search.score"]))
        return docs_score

    def import_docs(self,
                    output_path,
                    num_docs=1):
        """import_docs."""
        print(f"Getting top {num_docs} docs from Azure Cognitive Search config")
        results = self.client.search(
            search_text="",
            query_type="semantic",
            query_language="en-us",
            semantic_configuration_name=self.configuration_name,
            top=num_docs)
        os.makedirs(output_path, exist_ok=True)
        for idx, result in enumerate(results):
            print(result)
            with open(os.path.join(output_path, f"file_{idx}.txt"), "w+") as f:
                f.write(result[self.config.content_key])
        return


def validate_config(configuration, expected_keys):
    """validate_config."""
    missing_keys = []
    for key in expected_keys:
        if key not in configuration:
            print("Missing expected key {key} in configuration!")
            missing_keys.append(key)
    if len(missing_keys) != 0:
        raise ValueError(f"acs_config missing '{missing_keys}' fields."
                         + "They are required when using Azure Cognitive Search")
    return


def acs_client_from_config(acs_config, credential):
    """acs_client_from_config."""
    credential = credential
    return ACS.from_config(ACSConnection(
        credential=credential,
        endpoint=acs_config["endpoint"],
        index_name=acs_config["index_name"],
        configuration_name=acs_config.get("semantic_config_name", "default"),
        content_key=acs_config.get("content_key", "content"),
        source_key=acs_config.get("source_key", "sourcepage"),
        title_key=acs_config.get("title_key", "title"),
        api_version=acs_config.get('api_version', "2023-07-01-preview")))


def acs_existing_to_mlindex(acs_config, connection={}):
    """acs_existing_to_mlindex."""
    mlindex_config = {}
    if "content_key" not in acs_config:
        raise Exception("acs_config requires 'content_key' to be set for using existing ACS with embeddings")
    if "index_name" not in acs_config:
        raise Exception("acs_config requires 'index_name' to be set for using existing ACS with embeddings")
    if "embedding_key" not in acs_config:
        raise Exception("acs_config requires 'embedding_key' to be set for using existing ACS with embeddings")
    if 'api_version' not in acs_config:
        acs_config['api_version'] = "2023-07-01-preview"
        print(f"Using default api_version because none was specified: {acs_config['api_version']}")

    mlindex_config["index"] = {
        "kind": "acs",
        "engine": "azure-sdk",
        "index": acs_config.get('index_name'),
        "api_version": acs_config.get('api_version'),
        "field_mapping": {
            "content": acs_config.get("content_key"),
            "embedding": acs_config.get("embedding_key")
        }
    }

    model_connection_args = {}
    connection_id = os.environ.get('AZUREML_WORKSPACE_CONNECTION_ID_AOAI')
    if 'hugging_face' not in acs_config["embedding_model_uri"]:
        if connection_id is not None:
            model_connection_args['connection_type'] = 'workspace_connection'
            model_connection_args['connection'] = {'id': connection_id}
        else:
            if "open_ai" in acs_config["embedding_model_uri"]:
                ws = get_workspace_from_environment()
                connection_args["connection_type"] = "workspace_keyvault"
                connection_args["connection"] = {
                    "subscription": ws.subscription_id if ws is not None else "",
                    "resource_group": ws.resource_group if ws is not None else "",
                    "workspace": ws.name if ws is not None else "",
                    "key": "OPENAI-API-KEY"
                }

    embedding = EmbeddingsContainer.from_uri(acs_config["embedding_model_uri"], **model_connection_args)
    mlindex_config["embeddings"] = embedding.get_metadata()

    # Optionally set these other fields
    if 'source_key' in acs_config:
        mlindex_config["index"]["field_mapping"]["url"] = acs_config.get("source_key")
    if 'source_file_key' in acs_config:
        mlindex_config["index"]["field_mapping"]["filename"] = acs_config.get("source_file_key")
    if 'title_key' in acs_config:
        mlindex_config["index"]["field_mapping"]["title"] = acs_config.get("title_key")
    if 'metadata_key' in acs_config:
        mlindex_config["index"]["field_mapping"]["metadata"] = acs_config.get("metadata_key")

    if not isinstance(connection, DefaultAzureCredential):
        mlindex_config["index"] = {**mlindex_config["index"], **connection}

    # Keyvault auth and Default ambient auth need the endpoint, Workspace Connection auth could get endpoint.
    mlindex_config["index"]["endpoint"] = acs_config['endpoint']
    output = Path(args.ml_index)
    output.mkdir(parents=True, exist_ok=True)
    with open(output / "MLIndex", "w") as f:
        yaml.dump(mlindex_config, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--acs_config", type=str, required=True)
    parser.add_argument("--output_data", type=str, required=True)
    parser.add_argument("--ml_index", type=str, required=True)
    parser.add_argument("--num_docs", type=str, required=True)
    parser.add_argument("--use_existing", type=str, required=False, default=False)
    args = parser.parse_args()

    acs_config = json.loads(args.acs_config)
    print(f"Using acs_config: {json.dumps(acs_config, indent=2)}")
    validate_config(acs_config, ["index_name"])
    from azureml.core import Run
    run = Run.get_context()
    connection_id = os.environ.get('AZUREML_WORKSPACE_CONNECTION_ID_ACS', None)
    use_existing = args.use_existing == "True" or args.use_existing == "true"
    connection_args = {}
    if connection_id is not None:
        connection_args['connection_type'] = 'workspace_connection'
        connection_args['connection'] = {'id': connection_id}
        connection = get_connection_by_id_v2(connection_id)
        acs_config['endpoint'] = connection['properties']['target']
        acs_config['api_version'] = connection['properties'] \
            .get('metadata', {}) \
            .get('apiVersion', "2023-07-01-preview")
    elif 'endpoint_key_name' in acs_config:
        connection_args['connection_type'] = 'workspace_keyvault'
        ws = run.experiment.workspace
        connection_args['connection'] = {
            'key': acs_config['endpoint_key_name'],
            "subscription": ws.subscription_id,
            "resource_group": ws.resource_group,
            "workspace": ws.name,
        }

    credential = get_connection_credential(connection_args)
    client = acs_client_from_config(acs_config, credential)
    docs = client.import_docs(args.output_data, args.num_docs)
    if use_existing:
        acs_existing_to_mlindex(acs_config=acs_config, connection=connection_args)
