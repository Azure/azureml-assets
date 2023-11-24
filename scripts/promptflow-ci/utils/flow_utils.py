import copy
import json
import os
import random
import re
from datetime import datetime
from pathlib import Path
import shutil
from unittest import mock
import yaml

from utils.logging_utils import log_error, log_debug, log_warning


def create_flow_run_identifier(flow_id, flow_run_id):
    """Generate the global unique flow run identifier."""
    return f"{flow_id}:{flow_run_id}"


def resolve_flow_run_identifier(flow_run_identifier):
    """Resolve the flow run identifier to flow id and flow run id."""
    return flow_run_identifier.split(":")[0], flow_run_identifier.split(":")[1]


def _validate_meta(meta, flow_dir):
    if meta["type"] not in ["standard", "evaluate", "chat", "rag"]:
        raise ValueError(f"Unknown type in meta.json. model dir: {flow_dir}.")
    stage = meta["properties"]["promptflow.stage"]
    if stage not in ["test", "prod", "disabled"]:
        raise ValueError(f"Unknown stage in meta.json. flow dir: {flow_dir}.")


def _general_copy(src, dst, make_dirs=True):
    """Wrapped `shutil.copy2` function for possible "Function not implemented"
    exception raised by it.

    Background: `shutil.copy2` will throw OSError when dealing with Azure File.
    See https://stackoverflow.com/questions/51616058 for more information.
    """
    if make_dirs:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
    if hasattr(os, "listxattr"):
        with mock.patch("shutil._copyxattr", return_value=[]):
            shutil.copy2(src, dst)
    else:
        shutil.copy2(src, dst)


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise ValueError(f"Path {src} does not exist.")
    if src.is_file():
        _general_copy(src, dst)
    if src.is_dir():
        for name in src.glob("*"):
            _copy(name, dst / name.name)


def _assign_flow_values(flow_dirs, tmp_folder_path):
    """Assign the flow values and update flow.dag.yaml."""
    log_debug("\n=======Start overriding values for flows=======")
    updated_bulk_test_main_flows_dirs = []
    for flow_dir in flow_dirs:
        dst_path = (tmp_folder_path / flow_dir.parents[0].name).resolve()
        _copy(Path(flow_dir), dst_path)
        log_debug(dst_path)
        updated_bulk_test_main_flows_dirs.append(dst_path)

    for flow_dir in updated_bulk_test_main_flows_dirs:
        flow_dir_name = flow_dir.name
        flow_dir_name = flow_dir_name.replace("-", "_")
        script_flow_dir = Path(os.path.join(
            os.getcwd(), "scripts", "promptflow-ci", "test_configs", "flows", flow_dir_name))
        if not (script_flow_dir.exists() and (script_flow_dir / "values.json").exists()):
            log_warning(
                f"No available flow values found for '{flow_dir_name}'. Skip flow values assignment.")
            continue
        with open(os.path.join(script_flow_dir, "values.json"), "r") as fin:
            values = json.load(fin)
        with open(Path(flow_dir) / "flow.dag.yaml", "r") as dag_file:
            flow_dag = yaml.safe_load(dag_file)
        # Override connection/inputs in nodes
        if "nodes" in values:
            log_debug(f"Start overriding values for nodes for '{flow_dir.name}'.")
            for override_node in values["nodes"]:
                for flow_node in flow_dag["nodes"]:
                    if flow_node["name"] == override_node["name"]:
                        if "connection" in override_node:
                            # Override connection
                            flow_node["connection"] = override_node["connection"]
                        if "inputs" in override_node:
                            for input_name, input_value in override_node["inputs"].items():
                                # Override input
                                flow_node["inputs"][input_name] = input_value
            with open(flow_dir / "flow.dag.yaml", "w", encoding="utf-8") as dag_file:
                yaml.dump(flow_dag, dag_file, allow_unicode=True)
    log_debug("=======Complete overriding values for flows=======\n")
    return updated_bulk_test_main_flows_dirs


def _add_llm_node_variant(bulk_test_main_flows_dirs):
    log_debug("\n=======Start adding variants for flows=======")

    for bulk_test_main_flows_dir in bulk_test_main_flows_dirs:
        with open(Path(bulk_test_main_flows_dir) / "flow.dag.yaml", "r") as dag_file:
            flow_dag = yaml.safe_load(dag_file)
        nodes = flow_dag["nodes"]
        nodes_with_llm_tool = [(index, node) for index, node in enumerate(
            nodes) if node["type"] == "llm"]
        if len(nodes_with_llm_tool) == 0:
            log_warning(f"flow {Path(bulk_test_main_flows_dir).name} doesn't have nodes wit llm type tools. Skip adding llm "
                  f"node variant.")
            continue
        for index, node in nodes_with_llm_tool:
            if "use_variants" in node:
                log_debug(
                    f"flow {Path(bulk_test_main_flows_dir).name} already has variants. Skip adding llm node variant.")
                return
        # Select the first node to a one variant for it
        node_to_add_variant = copy.deepcopy(nodes_with_llm_tool[0][1])
        index_of_node_to_add_variant = nodes_with_llm_tool[0][0]
        nodes[index_of_node_to_add_variant] = {
            "name": node_to_add_variant["name"], "use_variants": True}
        # tool_name_for_added_variant = f"{node_to_add_variant['tool']}__variant_1"
        log_debug(f"Adding one variant for node {node_to_add_variant['name']} in bulk_test_main_flow "
              f"{Path(bulk_test_main_flows_dir).name}")
        node_variants = {}
        node_name = node_to_add_variant.pop("name")
        default_variant = {"node": node_to_add_variant}
        variants = {"variant_0": default_variant}
        # Add one variant for the node
        variants["variant_1"] = copy.deepcopy(default_variant)
        node_variants[node_name] = {
            "default_variant_id": "variant_0", "variants": variants}
        flow_dag["node_variants"] = node_variants
        with open(Path(bulk_test_main_flows_dir) / "flow.dag.yaml", "w", encoding="utf-8") as dag_file:
            yaml.dump(flow_dag, dag_file, allow_unicode=True)
    log_debug("=======Complete adding variants for flows=======\n")


def construct_create_flow_from_sample_payload(sample_name, flow_definition_file_path):
    return {"SampleName": sample_name, "FlowDefinitionFilePath": flow_definition_file_path}


def construct_create_flow_payload_of_new_contract(flow, flow_meta, properties):
    flow_type = flow_meta.get("type", None)
    if flow_type:
        mapping = {
            "standard": "default",
            "evaluate": "evaluation",
            "chat": "chat",
            "rag": "rag"
        }
        flow_type = mapping[flow_type]

    return {
        "flowName": flow_meta.get("display_name", None),
        "description": flow_meta.get("description", None),
        "tags": flow_meta.get("tags", None),
        "flowType": flow_type,
        "details": properties.get("promptflow.details.source", None) if properties else None,
        "flowRunSettings": {
            "batch_inputs": properties.get("update_promptflow.batch_inputs", None) if properties else None,
        },
        "flowDefinitionFilePath": flow.path,
        "isArchived": False,
    }


def construct_submit_flow_payload_of_new_contract(flow_id, batch_data_inputs, runtime_name, flow_dag, flow_submit_mode):
    flow_run_id = f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}_{random.randint(100000,999999)}"
    tuning_node_names = [node["name"]
                         for node in flow_dag["nodes"] if "use_variants" in node]
    submit_flow_payload = {
        "flowId": flow_id,
        "flowRunId": flow_run_id,
        "flowSubmitRunSettings": {
            "runtimeName": runtime_name,
            "runMode": "BulkTest",
            "batchDataInput": {"dataUri": batch_data_inputs},
            # Need to populate this field for the LLM node with variants
            "tuningNodeNames": tuning_node_names,
        },
        "asyncSubmission": True if flow_submit_mode == "async" else False,
        "useWorkspaceConnection": True,
        "useFlowSnapshotToSubmit": True,
    }
    return submit_flow_payload


def construct_submit_bulk_test_payload_of_new_contract(evaluation_flow_dag, evaluation_flow_resource_id,
                                                       batch_data_inputs, bulk_test_config, main_flow_id,
                                                       main_flow_dag, runtime_name, evaluation_flow_name,
                                                       main_flow_name):
    llm_nodes_in_evaluation_flow = [
        node for node in evaluation_flow_dag["nodes"] if node["type"] == "llm"]
    llm_node_connection_overrides = []
    for node in llm_nodes_in_evaluation_flow:
        connection_override = {
            "connectionSourceType": "Node",
            "nodeName": node["name"],
            "connectionName": "aoai_connection",  # Hard code for now
        }
        llm_node_connection_overrides.append(connection_override)

    evaluation_run_id = f"evaluate_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    evaluation_flow_run_settings = {
        "evaluation": {
            "flowRunDispalyName": f"{evaluation_flow_name}-bulktest",
            "connectionOverrides": llm_node_connection_overrides,
            "inputsMapping": bulk_test_config["evalFlowInputsMapping"],
            "flowRunId": evaluation_run_id,
            "batchDataInput": {"dataUri": batch_data_inputs},
        }
    }

    node_variants = main_flow_dag["node_variants"] if "node_variants" in main_flow_dag else {
    }
    tuning_node_names = [node["name"]
                         for node in main_flow_dag["nodes"] if "use_variants" in node]
    tuning_node_settings = {}
    for tuning_node_name in tuning_node_names:
        variant_ids = list(node_variants[tuning_node_name]["variants"].keys())
        tuning_node_settings.update(
            {tuning_node_name: {"variantIds": variant_ids}})

    bulk_test_payload = {
        "flowId": main_flow_id,
        "flowRunDisplayName": f"{main_flow_name}-bulktest",
        "flow": {
            "evaluationFlows": {"evaluation": {"referenceResourceId": evaluation_flow_resource_id}},
        },
        "flowRunId": f"run_{datetime.now().strftime('%Y%m%d%H%M%S')}",
        "flowSubmitRunSettings": {
            "runMode": "BulkTest",
            "batchDataInput": {"dataUri": batch_data_inputs},
            "runtimeName": runtime_name,
            "inputsMapping": bulk_test_config["mainFlowInputsMapping"],
            "tags": {},
            "description": f"CI triggered Bulk test & Evaluate for evaluation flow '{evaluation_flow_name}', "
            f"main flow: {main_flow_name} main flow ID: {main_flow_id}",
            "tuningNodeSettings": tuning_node_settings,
            "evaluationFlowRunSettings": evaluation_flow_run_settings,
        },
        "asyncSubmission": True,
        "useWorkspaceConnection": True,
        "useFlowSnapshotToSubmit": True,
    }
    return bulk_test_payload, evaluation_run_id


def resolve_bulk_test_config_of_new_contract(flow_dir: Path):
    with open(flow_dir / "bulk_test_config_of_new_contract.json", "r") as config:
        bulk_test_config = json.load(config)
    if (
        bulk_test_config is None
        or bulk_test_config["mainFlowInputsMapping"] is None
        or bulk_test_config["evalFlowInputsMapping"] is None
        or bulk_test_config["mainFlowFolderName"] is None
    ):
        raise Exception("Invalid bulk test config file.")

    main_flow_folder_name = bulk_test_config["mainFlowFolderName"].replace(
        "_", "-")
    return main_flow_folder_name, bulk_test_config


def construct_flow_link(aml_resource_uri, subscription, resource_group, workspace, experiment_id, flow_id, ux_flight):
    flow_link_format = (
        "{aml_resource_uri}/prompts/flow/{experiment_id}/{flow_id}/details?wsid=/subscriptions/"
        "{subscription}/resourceGroups/{resource_group}/providers/Microsoft.MachineLearningServices/"
        "workspaces/{workspace}&flight={ux_flight}"
    )
    return flow_link_format.format(
        aml_resource_uri=aml_resource_uri,
        subscription=subscription,
        resource_group=resource_group,
        workspace=workspace,
        experiment_id=experiment_id,
        flow_id=flow_id,
        ux_flight=ux_flight,
    )


def get_flow_link(create_flow_response_json, aml_resource_uri, subscription, resource_group, workspace, experiment_id,
                  ux_flight):
    flow_id = create_flow_response_json["flowId"]
    return construct_flow_link(aml_resource_uri, subscription, resource_group, workspace, experiment_id, flow_id,
                               ux_flight)


def get_flow_run_ids(bulk_test_response_json):
    bulk_test_id = bulk_test_response_json["bulkTestId"]
    flow_run_logs = bulk_test_response_json["flowRunLogs"]
    flow_run_ids = [run_id for run_id in list(
        flow_run_logs.keys()) if run_id != bulk_test_id]
    log_debug(f"flow_run_ids in utils: {flow_run_ids}")
    return flow_run_ids


def construct_flow_run_link(
    aml_resource_uri, subscription, resource_group, workspace, experiment_id, flow_id, flow_run_id
):
    bulk_test_run_link_format = (
        "{aml_resource_uri}/prompts/flow/{experiment_id}/{flow_id}/run/{flow_run_id}/details?wsid=/"
        "subscriptions/{subscription}/resourceGroups/{resource_group}/providers/"
        "Microsoft.MachineLearningServices/workspaces/{workspace}&flight=promptflow"
    )
    return bulk_test_run_link_format.format(
        aml_resource_uri=aml_resource_uri,
        subscription=subscription,
        resource_group=resource_group,
        workspace=workspace,
        experiment_id=experiment_id,
        flow_id=flow_id,
        flow_run_id=flow_run_id,
    )


def get_flow_run_link(
    bulk_test_response_json, aml_resource_uri, subscription, resource_group, workspace, experiment_id, flow_run_id
):
    flow_run_resource_id = bulk_test_response_json["flowRunResourceId"]
    flow_id, _ = _resolve_flow_run_resource_id(flow_run_resource_id)
    link = construct_flow_run_link(
        aml_resource_uri=aml_resource_uri,
        subscription=subscription,
        resource_group=resource_group,
        workspace=workspace,
        experiment_id=experiment_id,
        flow_id=flow_id,
        flow_run_id=flow_run_id,
    )
    return link


def _resolve_flow_run_resource_id(flow_run_resource_id):
    """Get flow id and flow run id from flow run resource id."""
    if flow_run_resource_id.startswith("azureml://"):
        flow_run_resource_id = flow_run_resource_id[len("azureml://"):]
    elif flow_run_resource_id.startswith("azureml:/"):
        flow_run_resource_id = flow_run_resource_id[len("azureml:/"):]

    pairs = re.findall(r"([^\/]+)\/([^\/]+)", flow_run_resource_id)
    flows = [pair for pair in pairs if pair[0] == "flows"]
    flow_runs = [pair for pair in pairs if pair[0] == "flowRuns"]
    if len(flows) == 0 or len(flow_runs) == 0:
        log_error(
            f"Resolve flow run resource id [{flow_run_resource_id}] failed")
        return None, None
    else:
        return flows[0][1], flow_runs[0][1]
