# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""File for the flow creation."""
import argparse
import os
import json
import errno
import uuid

import requests
import traceback
import time
from pathlib import Path
from typing import Tuple
from logging import Logger
from azureml.core import Run, Workspace
from azureml.core.run import _OfflineRun
from azureml.rag.utils.logging import (
    get_logger,
    enable_stdout_logging,
    enable_appinsights_logging,
    track_activity,
    _logger_factory,
)


logger = get_logger("flow_creation")

MAX_POST_TIMES = 3
SLEEP_DURATION = 1
USE_CODE_FIRST = True
USE_CHAT_FLOWS = True

SERVICE_ENDPOINT = os.environ.get("AZUREML_SERVICE_ENDPOINT", "")
EXPERIMENT_SCOPE = os.environ.get("AZUREML_EXPERIMENT_SCOPE", "")
WORKSPACE_SCOPE = os.environ.get("AZUREML_WORKSPACE_SCOPE", "")
RUN_TOKEN = os.environ.get("AZUREML_RUN_TOKEN", "")
CODE_DIR = "rag_code_flow"
_CITATION_TEMPLATE = r"\nPlease add citation after each sentence when possible in a form \"(Source: citation)\"."
_USER_INPUT = r"{{contexts}} \n user: {{question}} \nassistant:"
_MODIFY_INPUT = (
    r"Given the following conversation history and the users next question,"
    + r"rephrase the question to be a stand alone question.\nIf the conversation is irrelevant "
    + r"or empty, just restate the original question.\nDo not add more details than necessary to the question."
)
_CHAT_HISTORY = (
    r"\n chat history: \n{% for item in chat_history %} user: \n{{ item.inputs.question }} "
    + r"\nassistant: \n{{ item.outputs.output }} \n{% endfor %}"
)
_MODIFY_PROMPT = (
    r"system: \n"
    + _MODIFY_INPUT
    + r"\nconversation:\n"
    + _CHAT_HISTORY
    + r"\n\nuser:\nFollow up Input: {{question}} \nStandalone Question:"
)
_STATIC_METRIC_PRIORITY_LIST = ["gpt_similarity", "gpt_relevance", "bert_f1"]


def post_processing_prompts(prompt, citation_templates, user_input, is_chat):
    """Post processing prompts to include multiple roles to make it compatible with completion and chat API."""
    if is_chat:
        full_prompt = (
            r"system: \n"
            + prompt
            + citation_templates
            + r"\n\n user: \n {{contexts}} \n"
            + _CHAT_HISTORY
            + r"\nuser: {{question}} \nassistant:"
        )
    else:
        full_prompt = (
            r"system: \n" + prompt + citation_templates + r"\n\n user: \n " + user_input
        )

    return full_prompt


def get_default_headers(token, content_type=None, read_bytes=None):
    """Get default headers."""
    headers = {"Authorization": "Bearer %s" % token}

    if content_type:
        headers["Content-Type"] = content_type

    if read_bytes:
        headers["Content-Length"] = "%d" % len(read_bytes)

    return headers


def get_workspace_and_run() -> Tuple[Workspace, Run]:
    """get_workspace_and_run."""
    run = Run.get_context()
    if isinstance(run, _OfflineRun):  # for local testing
        workspace = Workspace.from_config()
    else:
        workspace = run.experiment.workspace
    return workspace, run


def try_request(url, json_payload, headers, activity_logger):
    """Try to send request.

    It will sallow exceptions to avoid breaking callers.
    """
    activity_logger.info("[Promptflow Creation]: url is: {}".format(url))
    for index in range(MAX_POST_TIMES):
        try:
            resp = requests.post(url, json=json_payload, headers=headers)
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as http_err:
            if index == 0:
                activity_logger.info(
                    "[Promptflow Creation]: Failed to send json payload log {}".format(
                        json_payload
                    )
                )
            activity_logger.info(
                "[Promptflow Creation]: Failed to send log with error {} response Code: {}, "
                "Content: {}. Detail: {}".format(
                    str(http_err),
                    resp.status_code,
                    resp.content,
                    traceback.format_exc(),
                )
            )

            if resp.status_code >= 500:
                time.sleep(SLEEP_DURATION)
                print("Retrying...")
            else:
                return resp
        except BaseException as exc:  # pylint: disable=broad-except
            activity_logger.info(
                "[Promptflow Creation]: Failed to send log: {} with error {}. Detail: {}.".format(
                    json_payload, exc, traceback.format_exc()
                )
            )

            return resp


def json_stringify(s):
    """json_stringify."""
    s = json.dumps(s)
    if s.startswith('"'):
        s = s[1:]
    if s.endswith('"'):
        s = s[:-1]
    return s


def get_connection_name(s):
    """get_connection_name."""
    parsed_connection = s.split("/connections/", 1)
    connection_name = parsed_connection[1]
    return connection_name


def get_deployment_and_model_name(s):
    """get_deployment_and_model_name."""
    model_type = s.split("://")[0]
    if model_type == "open_ai":
        model_name = s.split("/model/")[1]
        return (model_name, model_name)
    else:
        deployment_and_model = s.split("/deployment/")[1]
        deployment_name = deployment_and_model.split("/model/")[0]
        model_name = deployment_and_model.split("/model/")[1]
        return (deployment_name, model_name)


def upload_code_files(ws: Workspace):
    """Upload the files in the code flow directory."""
    from azureml.data.dataset_factory import FileDatasetFactory

    working_directory = ws.datastores.get("workspaceworkingdirectory")
    asset_id = str(uuid.uuid4())
    dest = f"index-in-pf-examples/Promptflows/{asset_id}/{CODE_DIR}"
    FileDatasetFactory.upload_directory(
        os.path.join(Path(__file__).parent.absolute(), CODE_DIR),
        (working_directory, dest),
    )

    return dest


def main(args, ws, current_run, activity_logger: Logger):
    """Extract main method."""
    activity_logger.info(
        "[Promptflow Creation]: Received and parsed arguments for promptflow creation in RAG."
        + " Creating promptflow now..."
    )
    print("best_prompts: %s" % args.best_prompts)
    print("mlindex_name: %s" % args.mlindex_name)
    print("mlindex_asset_id: %s" % args.mlindex_asset_id)
    print("llm_connection_name: %s" % args.llm_connection_name)
    print("llm_config: %s" % args.llm_config)
    print("embedding_connection: %s" % args.embedding_connection)
    print("embeddings_model: %s" % args.embeddings_model)

    # parse completion/chat: connection, deployment name, model name
    # parse embedding: connection, deployment name, model name
    completion_connection_name = get_connection_name(args.llm_connection_name)
    completion_config = json.loads(args.llm_config)

    if completion_config.get("type") == "open_ai":
        completion_model_name = completion_config.get("model_name", "gpt-3.5-turbo")
        completion_deployment_name = completion_model_name
        completion_model_name = (
            "gpt-3.5-turbo" if completion_model_name is None else completion_model_name
        )
        completion_deployment_name = (
            "gpt-3.5-turbo"
            if completion_deployment_name is None
            else completion_deployment_name
        )
        completion_provider = "OpenAI"
    else:
        completion_model_name = completion_config.get("model_name", "gpt-35-turbo")
        # If completion deployment not set, set as blank in PF rather than a placeholder
        completion_deployment_name = completion_config.get("deployment_name", None)
        # Set default if key exsits but is set to None (as it is for basic pipelines)
        completion_model_name = (
            "gpt-35-turbo" if completion_model_name is None else completion_model_name
        )
        completion_deployment_name = (
            "" if completion_deployment_name is None else completion_deployment_name
        )
        completion_provider = "AzureOpenAI"

    embedding_connection_name = get_connection_name(args.embedding_connection)
    if (
        completion_connection_name == "azureml-rag-default-aoai"
        and embedding_connection_name != "azureml-rag-default-aoai"
    ):
        # default completion connection name to embedding ones if embedding conenction is provided
        completion_connection_name = embedding_connection_name

    embedding_deployment_name_and_model_name = get_deployment_and_model_name(
        args.embeddings_model
    )
    embedding_deployment_name = embedding_deployment_name_and_model_name[0]
    embedding_model_name = embedding_deployment_name_and_model_name[1]

    print("completion_connection_name: %s" % completion_connection_name)
    print("completion_provider: %s" % completion_provider)
    print("completion_model_name: %s" % completion_model_name)
    print("completion_deployment_name: %s" % completion_deployment_name)

    print("embedding_connection_name: %s" % embedding_connection_name)
    print("embedding_deployment_name: %s" % embedding_deployment_name)
    print("embedding_model_name: %s" % embedding_model_name)

    if completion_model_name.startswith("gpt-") and USE_CHAT_FLOWS:
        print("Using chat flows")
        is_chat = True
        prefix = "chat_"
    else:
        print("Not using chat flows")
        is_chat = False
        prefix = ""

    if args.best_prompts is None:
        if is_chat:
            top_prompts = [
                "You are an AI assistant that helps users answer questions given a specific context and "
                + "conversation history. You will be given a context and chat history, and then asked a "
                + "question based on that context and history. Your answer should be as "
                + "precise as possible, and should only come from the context.",
                "You are an AI assistant that helps users answer questions given a specific context and "
                + "chat history. You will be given a context and history and asked a question based on "
                + "that context and chat history. Your answer should be as precise as possible and should "
                + "only come from the context.",
                "You are an chat assistant for helping users answering question given a specific context and "
                + "history. You are given a context and conversation history and you'll be asked a question "
                + "based on the context and history. Your answer should be as precise as possible and answer "
                + "should be only from the context.",
            ]
        else:
            top_prompts = [
                "You are an AI assistant that helps users answer questions given a specific context. You will be "
                + "given a context, and then asked a question based on that context. Your answer should be as "
                + "precise as possible, and should only come from the context.",
                "You are an AI assistant that helps users answer questions given a specific context. You will be "
                + "given a context and asked a question based on that context. Your answer should be as precise "
                + "as possible and should only come from the context.",
                "You are an chat assistant for helping users answering question given a specific context.You are "
                + "given a context and you'll be asked a question based on the context.Your answer should be "
                + "as precise as possible and answer should be only from the context.",
            ]
    else:
        with open(args.best_prompts, "r") as f:
            promt_json = json.load(f)
            top_prompts = promt_json.get("best_prompt")
            if top_prompts is None:
                # the output is in metrics mode:
                top_prompts = []
                for metric_name in _STATIC_METRIC_PRIORITY_LIST:
                    top_prompts_for_metric = promt_json.get(
                        "best_prompt_" + metric_name
                    )
                    if top_prompts_for_metric is not None:
                        top_prompts.append(top_prompts_for_metric[0])

    if args.mlindex_asset_id is None:
        activity_logger.info("[Promptflow Creation]: No MlIndex asset Id passed in.")
        raise FileNotFoundError(
            errno.ENOENT, os.strerror(errno.ENOENT), "mlindex_asset_id_file"
        )
    else:
        with open(args.mlindex_asset_id, "r") as f:
            mlindex_asset_id = f.read()

    print(mlindex_asset_id)
    print(top_prompts)

    if isinstance(top_prompts, str):
        top_prompts = [top_prompts, top_prompts, top_prompts]

    if USE_CODE_FIRST:
        file_name = os.path.join(
            Path(__file__).parent.absolute(), "flow_yamls", prefix + "flow.dag.yaml"
        )
    else:
        file_name = os.path.join(
            Path(__file__).parent.absolute(),
            "flow_jsons",
            prefix + "flow_with_variants_mlindex.json",
        )

    # for top prompts, construct top templates and inject into variants
    with open(file_name, "r") as file:
        flow_with_variants = file.read()

    flow_name = args.mlindex_name + "-sample-flow"

    # Replace these values in both code first and json
    flow_with_variants = flow_with_variants.replace(
        "@@Embedding_Deployment_Name", embedding_deployment_name
    )
    flow_with_variants = flow_with_variants.replace(
        "@@Embedding_Connection", embedding_connection_name
    )
    flow_with_variants = flow_with_variants.replace(
        "@@Completion_Deployment_Name", completion_deployment_name
    )
    flow_with_variants = flow_with_variants.replace(
        "@@Completion_Connection", completion_connection_name
    )
    flow_with_variants = flow_with_variants.replace(
        "@@Completion_Provider", completion_provider
    )
    flow_with_variants = flow_with_variants.replace(
        "@@MLIndex_Asset_Id", mlindex_asset_id
    )

    api_name = "chat" if completion_model_name.startswith("gpt-") else "completion"
    flow_with_variants = flow_with_variants.replace("@@API", api_name)

    if USE_CODE_FIRST:
        # write flow dag yaml back to file
        with open(
            os.path.join(Path(__file__).parent.absolute(), CODE_DIR, "flow.dag.yaml"),
            "w",
        ) as file:
            file.write(flow_with_variants)
        import codecs

        # Write prompt file content for Variants
        for idx in range(0, len(top_prompts)):
            prompt_str = post_processing_prompts(
                json_stringify(top_prompts[idx]),
                _CITATION_TEMPLATE,
                _USER_INPUT,
                is_chat,
            )
            with open(
                os.path.join(
                    Path(__file__).parent.absolute(),
                    CODE_DIR,
                    f"Prompt_variants__Variant_{idx}.jinja2",
                ),
                "w",
            ) as file:
                file.write(codecs.decode(prompt_str, "unicode_escape"))

        if USE_CHAT_FLOWS:
            # Write extra modify query prompt to folder
            with open(
                os.path.join(
                    Path(__file__).parent.absolute(),
                    CODE_DIR,
                    "modify_query_with_history.jinja2",
                ),
                "w",
            ) as file:
                file.write(codecs.decode(_MODIFY_PROMPT, "unicode_escape"))

        # upload code
        yaml_path = upload_code_files(ws) + "/flow.dag.yaml"
        activity_logger.info(
            "[Promptflow Creation]: Code first flow files successfully uploaded!"
        )
        # Load in Json
        json_name = os.path.join(
            "flow_jsons", prefix + "flow_with_variants_mlindex_code_first.json"
        )
        with open(
            os.path.join(Path(__file__).parent.absolute(), json_name), "r"
        ) as file:
            flow_submit_data = file.read()

        flow_submit_data = flow_submit_data.replace("@@Flow_Name", flow_name)
        flow_submit_data = flow_submit_data.replace("@@Flow_Definition_Path", yaml_path)
        # replace values as it should
        activity_logger.info(
            "[Promptflow Creation]: Code first json payload successfully generated, trying to parse into json dict..."
        )
        json_payload = json.loads(flow_submit_data)
        activity_logger.info(
            "[Promptflow Creation]: Code first json payload successfully parsed, submit to Promptflow service now..."
        )
    else:
        flow_with_variants = flow_with_variants.replace("@@Flow_Name", flow_name)

        # replace variants with actual metric name and value
        flow_with_variants = flow_with_variants.replace(
            "@@prompt_variant_0",
            post_processing_prompts(
                json_stringify(top_prompts[0]), _CITATION_TEMPLATE, _USER_INPUT, is_chat
            ),
        )
        flow_with_variants = flow_with_variants.replace(
            "@@prompt_variant_1",
            post_processing_prompts(
                json_stringify(top_prompts[1]), _CITATION_TEMPLATE, _USER_INPUT, is_chat
            ),
        )
        flow_with_variants = flow_with_variants.replace(
            "@@prompt_variant_2",
            post_processing_prompts(
                json_stringify(top_prompts[2]), _CITATION_TEMPLATE, _USER_INPUT, is_chat
            ),
        )

        if USE_CHAT_FLOWS:
            flow_with_variants = flow_with_variants.replace(
                "@@modify_prompt", _MODIFY_PROMPT
            )

        activity_logger.info(
            "[Promptflow Creation]: Json payload successfully generated, trying to parse into json dict..."
        )
        json_payload = json.loads(flow_with_variants)
        activity_logger.info(
            "[Promptflow Creation]: Json payload successfully parsed, submit to promptflow service now..."
        )

    ###########################################################################
    # ### construct PF MT service endpoints ### #
    promptflow_mt_url = SERVICE_ENDPOINT + "/flow/api" + WORKSPACE_SCOPE + "/flows"
    headers = get_default_headers(RUN_TOKEN, content_type="application/json")

    response = try_request(promptflow_mt_url, json_payload, headers, activity_logger)
    pf_response_json = json.loads(response.text)
    if "flowResourceId" not in pf_response_json:
        activity_logger.error(
            "[Promptflow Creation]: Flow creation failed with Response Code: {response.status_code}, response:{response.text}."
        )
    else:
        flow_id = pf_response_json["flowResourceId"]
        activity_logger.info(
            "[Promptflow Creation]: Flow creation Succeeded! Id is:" + flow_id,
            extra={"properties": {"flow_id": flow_id}},
        )
        run_properties = current_run.get_properties()
        parent_run_id = run_properties["azureml.pipelinerunid"]
        parent_run = ws.get_run(parent_run_id)
        parent_run.add_properties({"azureml.promptFlowResourceId": flow_id})
        activity_logger.info(
            "[Promptflow Creation]: Add into run property Succeed! All operation is done!",
            extra={"properties": {"flow_id": flow_id}},
        )


def main_wrapper(args, ws, current_run, logger):
    """Wrap around main function."""
    with track_activity(
        logger, "flow_creation", custom_dimensions={"llm_config": args.llm_config}
    ) as activity_logger:
        try:
            main(args, ws, current_run, activity_logger)
        except Exception:
            # activity_logger doesn't log traceback
            activity_logger.error(
                "[Promptflow Creation]: Failed with exception:" + traceback.format_exc()
            )
            raise


if __name__ == "__main__":
    enable_stdout_logging()
    enable_appinsights_logging()

    # TODO: add actual implementation to create a PF flow from a json payload from previous step
    print("Creating prompt flow")

    parser = argparse.ArgumentParser()
    parser.add_argument("--best_prompts", type=str)
    parser.add_argument("--mlindex_asset_id", type=str)
    parser.add_argument(
        "--mlindex_name",
        type=str,
        required=False,
        dest="mlindex_name",
        default="Baker_Example_With_Variants",
    )
    parser.add_argument(
        "--llm_connection_name",
        type=str,
        required=False,
        dest="llm_connection_name",
        default="/subscriptions/dummy/resourceGroups/dummy/providers/"
        + "Microsoft.MachineLearningServices/workspaces/dummy/connections/azureml-rag-default-aoai",
    )
    parser.add_argument(
        "--llm_config",
        type=str,
        required=False,
        dest="llm_config",
        default='{"type": "azure_open_ai", "model_name": "gpt-35-turbo", '
        + '"deployment_name": "gpt-35-turbo", "temperature": 0, "max_tokens": 2000}',
    )
    parser.add_argument(
        "--embedding_connection",
        type=str,
        required=False,
        dest="embedding_connection",
        default="/subscriptions/dummy/resourceGroups/dummy/providers/"
        + "Microsoft.MachineLearningServices/workspaces/dummy/connections/azureml-rag-default-aoai",
    )
    parser.add_argument(
        "--embeddings_model",
        type=str,
        required=False,
        dest="embeddings_model",
        default="azure_open_ai://endpoint/dummy/deployment/text-embedding-ada-002/model/text-embedding-ada-002",
    )

    args = parser.parse_args()
    ws, current_run = get_workspace_and_run()

    try:
        main_wrapper(args, ws, current_run, logger)
    finally:
        if _logger_factory.appinsights:
            _logger_factory.appinsights.flush()
            time.sleep(5)  # wait for appinsights to send telemetry
