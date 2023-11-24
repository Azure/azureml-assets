import argparse
import os
from pathlib import Path
import yaml
import tempfile
import json
import shutil
from markdown import markdown
from bs4 import BeautifulSoup
import time
import copy
from azureml.core import Workspace

from utils.utils import get_diff_files
from utils.logging_utils import log_debug, log_error, log_warning, debug_output
from azure.storage.blob import ContainerClient
from utils.mt_client import get_mt_client
from promptflow.azure import PFClient
from azure.identity import DefaultAzureCredential
from utils import flow_utils
from promptflow.azure._load_functions import load_flow
from promptflow.azure._utils.gerneral import is_remote_uri


TEST_FOLDER = "test"
MODEL_FILE = "model.yaml"
MODELS_ROOT = "assets/promptflow/models/"


def validate_downlaod(model_dir):
    yaml_file = os.path.join(model_dir, MODEL_FILE)
    with open(yaml_file, 'r') as file:
        model_yaml = yaml.safe_load(file)

    path = model_yaml.get("path")
    container_name = path.get("container_name")
    container_path = path.get("container_path")
    storage_name = path.get("storage_name")
    download_blob_to_file(container_name, container_path,
                          storage_name, os.path.join(model_dir, TEST_FOLDER))


def download_blob_to_file(container_name, container_path, storage_name, target_dir):
    account_url = f"https://{storage_name}.blob.core.windows.net"
    log_debug(f"\nCurrent storage_name: {storage_name}")
    log_debug(f"Current container_name: {container_name}")
    # Create the BlobServiceClient object
    # Because container has been enabled anonymous access, the ContainerClient should not contain credential.
    # Refer to https://stackoverflow.com/questions/60830598/getting-authorizationpermissionmismatch-on-a-public-container
    container_client = ContainerClient(account_url, container_name)

    log_debug(f"Downloading blobs under path {container_path} to {target_dir}")
    blob_list = container_client.list_blobs(container_path)
    for blob in blob_list:
        relative_path = os.path.relpath(blob.name, container_path)
        download_file_path = os.path.join(target_dir, relative_path)
        log_debug(f"Downloading blob {blob.name} to {download_file_path}")
        if not os.path.exists(os.path.dirname(download_file_path)):
            log_debug(f"Make dir {os.path.dirname(download_file_path)}")
            os.makedirs(os.path.dirname(download_file_path))
        with open(download_file_path, "wb") as my_blob:
            download_stream = container_client.download_blob(blob.name)
            my_blob.write(download_stream.readall())


def get_changed_models(diff_files):
    changed_models = [os.path.join(MODELS_ROOT, "ask-wikipedia"), os.path.join(
        MODELS_ROOT, "web-classification")]  # this is main flow folder for eval
    deleted_models_path = []
    for file_path in diff_files:
        file_path_list = file_path.split("\t")
        git_diff_file_path = file_path_list[-1]
        if git_diff_file_path.startswith(MODELS_ROOT):
            if not os.path.exists(os.path.join(MODELS_ROOT, git_diff_file_path.split("/")[-2])) \
                    or git_diff_file_path == " ":
                deleted_models_path.append(file_path)
            else:
                changed_models.append(os.path.join(
                    MODELS_ROOT, git_diff_file_path.split("/")[-2]))

    log_debug(
        f"Find {len(deleted_models_path)} deleted models: {deleted_models_path}.")
    log_debug(f"Find {len(changed_models)} changed models: {changed_models}.")

    return changed_models


def get_all_models():
    all_models = [os.path.join(MODELS_ROOT, model)
                  for model in os.listdir(MODELS_ROOT)]
    log_debug(f"Find {len(all_models)} models: {all_models}.")

    return all_models


def _dump_workspace_config(subscription_id, resource_group, workspace_name):
    workspace_config = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name
    }
    config_path = os.path.join(os.path.abspath('.'), "config.json")
    with open(config_path, 'w') as config_file:
        config_file.write(json.dumps(workspace_config, indent=4))


def create_flows(flows_dirs):
    # key: flow dir name, value: (flow graph, flow create result, flow type)
    flows_creation_info = {}
    flow_validation_errors = []
    for flow_dir in flows_dirs:
        log_debug(f"\nChecking flow dir: {flow_dir}")
        with open(Path(flow_dir) / "flow.dag.yaml", "r") as dag_file:
            flow_dag = yaml.safe_load(dag_file)
        with open(Path(flow_dir) / "flow.meta.yaml", "r") as meta_file:
            flow_meta = yaml.safe_load(meta_file)
        flow_type = flow_meta["type"]

        flow_utils._validate_meta(flow_meta, flow_dir)

        # check if the flow.dag.yaml exits
        if not os.path.exists(Path(flow_dir) / "flow.dag.yaml"):
            log_warning(
                f"flow.dag.yaml not found in {flow_dir}. Skip this flow.")
            continue

        section_type = "gallery"
        if "properties" in flow_meta.keys() and "promptflow.section" in flow_meta["properties"].keys():
            section_type = flow_meta["properties"]["promptflow.section"]
        # check if the README.md exits
        # skip checking README exists due to template flows don't have README.md.
        if section_type != "template":
            if not os.path.exists(Path(flow_dir) / "README.md"):
                flow_validation_errors.append(
                    f"README.md not found in {flow_dir}. Please add README.md to the flow.")
                continue
            else:
                # Check Links in Markdown Files of Flows,
                # make sure it opens a new browser tab instead of refreshing the current page.
                def extract_links_from_file(file_path):
                    with open(file_path, "r") as file:
                        content = file.read()
                        html = markdown(content)
                        soup = BeautifulSoup(html, "html.parser")
                        return soup

                def check_links(soup):
                    valid_links = True
                    links = soup.find_all("a")
                    for link in links:
                        if link.get("target") != "_blank":
                            log_debug(f'Invalid link syntax: {link}')
                            valid_links = False
                    return valid_links

                readme_file = os.path.join(flow_dir, "README.md")
                log_debug(f"Checking links in {readme_file}")
                soup = extract_links_from_file(readme_file)
                valid_links = check_links(soup)
                if not valid_links:
                    flow_validation_errors.append(
                        f"Some links in {flow_dir}'s README file do not follow the required syntax. "
                        "To ensure that links in the flow's README file open in a new browser tab "
                        "instead of refreshing the current page when users view the sample introduction, "
                        "please use the following syntax: <a href='http://example.com' target='_blank'>link text</a>."
                    )
                    continue
        # Call MT to create flow
        log_debug(
            f"Starting to create/update flow. Flow dir: {Path(flow_dir).name}.")
        flow = load_flow(source=flow_dir)
        properties = flow_meta.get("properties", None)
        if properties and "promptflow.batch_inputs" in properties:
            input_path = properties["promptflow.batch_inputs"]
            samples_file = Path(flow_dir) / input_path
            if samples_file.exists():
                with open(samples_file, "r", encoding="utf-8") as fp:
                    properties["update_promptflow.batch_inputs"] = json.loads(
                        fp.read())

        flow_operations._resolve_arm_id_or_upload_dependencies_to_file_share(
            flow)
        log_debug(f"FlowDefinitionFilePath: {flow.path}")

        create_flow_payload = flow_utils.construct_create_flow_payload_of_new_contract(
            flow, flow_meta, properties)
        debug_output(create_flow_payload, "create_flow_payload",
                     Path(flow_dir).name, args.local)
        create_flow_result = mt_client.create_or_update_flow(
            create_flow_payload)
        experiment_id = create_flow_result["experimentId"]
        flows_creation_info.update({Path(flow_dir).name: (
            flow_dag, create_flow_result, flow_type, section_type)})

        if create_flow_result['flowId'] is None:
            raise Exception(
                f"Flow id is None when creating/updating mode {flow_dir}. Please make sure the flow is valid")
        debug_output(create_flow_result, "create_flow_result",
                     Path(flow_dir).name, args.local)
        flow_link = flow_utils.get_flow_link(create_flow_result, ux_endpoint, args.subscription_id,
                                             args.resource_group, args.workspace_name, experiment_id, args.ux_flight)
        log_debug(f"Flow link to Azure Machine Learning Portal: {flow_link}")

    if len(flow_validation_errors) > 0:
        log_debug(
            "Promptflow CI failed due to the following flow validation errors:", True)
        for failure in flow_validation_errors:
            log_error(failure)
        exit(1)
    return flows_creation_info


def _resolve_data_to_asset_id(test_data):
    from azure.ai.ml._artifacts._artifact_utilities import _upload_and_generate_remote_uri
    from azure.ai.ml.constants._common import AssetTypes

    def _get_data_type(_data):
        if os.path.isdir(_data):
            return AssetTypes.URI_FOLDER
        else:
            return AssetTypes.URI_FILE

    if is_remote_uri(test_data):
        # Pass through ARM id or remote url
        return test_data

    if os.path.exists(test_data):  # absolute local path, upload, transform to remote url
        data_type = _get_data_type(test_data)
        test_data = _upload_and_generate_remote_uri(
            run_operations._operation_scope,
            run_operations._datastore_operations,
            test_data,
            datastore_name=run_operations._workspace_default_datastore,
            show_progress=run_operations._show_progress,
        )
        if data_type == AssetTypes.URI_FOLDER and test_data and not test_data.endswith("/"):
            test_data = test_data + "/"
    else:
        raise ValueError(
            f"Local path {test_data!r} not exist. "
            "If it's remote data, only data with azureml prefix or remote url is supported."
        )
    return test_data


def check_flow_run_status(flow_runs_to_check, submitted_run_identifiers, check_run_status_interval, check_run_status_max_attempts):
    for flow_run_identifier in flow_runs_to_check:
        flow_id, flow_run_id = flow_utils.resolve_flow_run_identifier(
            flow_run_identifier)
        flow_run_link = flow_utils.construct_flow_run_link(ux_endpoint, args.subscription_id,
                                                           args.resource_group, args.workspace_name,
                                                           experiment_id, flow_id, flow_run_id)
        log_debug(f"Start checking flow run {flow_run_id} run, link to Azure Machine Learning Portal: "
                  f"{flow_run_link}")
        current_attempt = 0
        while current_attempt < check_run_status_max_attempts:
            bulk_test_run = run_workspace.get_run(run_id=flow_run_id)
            if bulk_test_run.status == "Completed":
                submitted_run_identifiers.remove(flow_run_identifier)
                break
            elif bulk_test_run.status == "Failed":
                submitted_run_identifiers.remove(flow_run_identifier)
                failed_flow_runs.update({flow_run_identifier: flow_run_link})
                break
            elif bulk_test_run.status == "Canceled":
                submitted_run_identifiers.remove(flow_run_identifier)
                failed_flow_runs.update({flow_run_identifier: flow_run_link})
                break

            current_attempt += 1
            time.sleep(check_run_status_interval)


def get_bulk_test_main_flows_dirs(flows_root, flow_dirs):
    bulk_test_main_flows_dirs = []
    for flow_dir in flow_dirs:
        flow_dir = Path(flow_dir).name.replace("-", "_")
        script_flow_dir = Path(os.path.join(
            os.getcwd(), "scripts", "promptflow-ci", "test_configs", "flows", flow_dir))
        if (script_flow_dir / "bulk_test_config_of_new_contract.json").exists():
            with open(script_flow_dir / "bulk_test_config_of_new_contract.json", 'r') as config:
                bulk_test_config = json.load(config)
                main_flow_folder_name = bulk_test_config["mainFlowFolderName"].replace(
                    "_", "-")
            if os.path.join(flows_root, main_flow_folder_name) not in bulk_test_main_flows_dirs:
                bulk_test_main_flows_dirs.append(
                    os.path.join(flows_root, main_flow_folder_name))

    return bulk_test_main_flows_dirs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant_id", type=str)
    parser.add_argument("--client_id", type=str)
    parser.add_argument("--client_secret", type=str)
    parser.add_argument("--subscription_id", type=str,
                        default="96aede12-2f73-41cb-b983-6d11a904839b")
    parser.add_argument("--resource_group", type=str, default="promptflow")
    parser.add_argument("--workspace_name", type=str,
                        default="chjinche-pf-eus")
    parser.add_argument("--ux_endpoint", type=str,
                        default="https://int.ml.azure.com")
    parser.add_argument("--ux_flight", type=str,
                        default="promptfilestorage,PFPackageTools,PFRunList,PromptBatchRunDesignV2,PFSourceRun")
    parser.add_argument("--mt_service_route", type=str,
                        default="https://eastus.api.azureml.ms/flow")
    parser.add_argument('--flow_submit_mode', type=str, default="sync")
    parser.add_argument('--run_time', type=str, default="default-mir")
    parser.add_argument('--skipped_flows', type=str, default="langchain_auto_gpt,bing_grounded_qna,bing_grounded_"
                                                             "chatgpt")
    parser.add_argument(
        "--local",
        help="local debug mode, will use interactive login authentication, and output the request "
        "response to local files",
        action='store_true'
    )
    args = parser.parse_args()

    # Get changed models folder or all models folder
    diff_files = get_diff_files()
    diff_files_list = {path.split('/')[-1] for path in diff_files}
    log_debug(f"Git diff files include:{diff_files_list}")

    if "promptflow_ci.py" in diff_files_list or "promptflow-ci.yml" in diff_files_list:
        log_debug("promptflow_ci.py or promptflow_ci.yml changed, test all models.")
        changed_models = get_all_models()
    else:
        changed_models = get_changed_models(diff_files)
        if len(changed_models) == 0:
            log_error(f"No change in {MODELS_ROOT}, skip flow testing.")
            exit(0)

    # Check download models
    errors = []
    for model_dir in changed_models:
        try:
            validate_downlaod(model_dir)
        except Exception as e:
            errors.append(e)

    if len(errors) > 0:
        log_error(f"Found {len(errors)} errors when downloading models")
        for error in errors:
            log_error(error)
        exit(1)

    # Check run flows
    handled_failures = []
    bulk_test_main_flows_dirs = get_bulk_test_main_flows_dirs(
        MODELS_ROOT, changed_models)
    bulk_test_main_flows_dirs = list(dict.fromkeys(bulk_test_main_flows_dirs))
    flows_dirs = [
        dir for dir in changed_models if dir not in bulk_test_main_flows_dirs]
    bulk_test_main_flows_dirs = [
        model_dir for model_dir in bulk_test_main_flows_dirs]

    # Filter out skipped flows
    if args.skipped_flows != "":
        skipped_flows = args.skipped_flows.split(",")
        log_debug(f"Skipped flows: {skipped_flows}")
        flows_dirs = [flow_dir for flow_dir in flows_dirs if Path(
            flow_dir).name not in skipped_flows]
        bulk_test_main_flows_dirs = [flow_dir for flow_dir in bulk_test_main_flows_dirs if Path(flow_dir).name not in
                                     skipped_flows]
    flows_dirs = [Path(os.path.join(dir, TEST_FOLDER)) for dir in flows_dirs]
    bulk_test_main_flows_dirs = [Path(os.path.join(
        model_dir, TEST_FOLDER)) for model_dir in bulk_test_main_flows_dirs]

    log_debug(flows_dirs)
    log_debug(bulk_test_main_flows_dirs)
    if len(flows_dirs) + len(bulk_test_main_flows_dirs) == 0:
        print("No flow code change, skip flow testing.")
        exit(0)

    ux_endpoint = args.ux_endpoint
    runtime_name = args.run_time

    mt_client = get_mt_client(
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
        args.tenant_id,
        args.client_id,
        args.client_secret,
        args.local,
        args.mt_service_route
    )

    _dump_workspace_config(args.subscription_id,
                           args.resource_group, args.workspace_name)
    credential = DefaultAzureCredential(additionally_allowed_tenants=[
                                        "*"], exclude_shared_token_cache_credential=True)
    pf_client = PFClient.from_config(credential=credential)
    flow_operations = pf_client._flows
    run_operations = pf_client._runs

    # region: Step1. create/update flows, and store flow creation info in flows_creation_info
    try:
        # add node variant for llm node
        tmp_folder_path = Path(tempfile.mkdtemp())
        log_debug(f"tmp folder path: {tmp_folder_path}")
        flows_dirs = flow_utils._assign_flow_values(
            flows_dirs, tmp_folder_path)
        bulk_test_main_flows_dirs = flow_utils._assign_flow_values(
            bulk_test_main_flows_dirs, tmp_folder_path)
        flow_utils._add_llm_node_variant(bulk_test_main_flows_dirs)

        flows_creation_info = create_flows(flows_dirs)
        bulk_test_main_flows_creation_info = create_flows(
            bulk_test_main_flows_dirs)
    except Exception as e:
        log_error("Error when creating flow")
        raise e
    finally:
        shutil.rmtree(tmp_folder_path)
    # endregion

    # region: Step2. submit bulk test runs and evaluation flow bulk test runs asynchronously based on the
    # flows_creation_info
    submitted_flow_run_identifiers = set()
    submitted_bulk_test_run_identifiers = set()
    submit_interval = 2  # seconds

    for flow_dir_name, creation_info in flows_creation_info.items():
        time.sleep(submit_interval)
        flow_dir = Path(os.path.join(MODELS_ROOT, flow_dir_name, TEST_FOLDER))
        flow_dag = creation_info[0]
        flow_create_result = creation_info[1]
        flow_type = creation_info[2]
        section_type = creation_info[3]
        flow_id = flow_create_result['flowId']
        flow_resource_id = flow_create_result["flowResourceId"]
        flow_name = flow_create_result["flowName"]
        # script flow dir contains flow input/param assignments of the flow,
        # which should be separated from promptflow model
        flow_dir_name = flow_dir_name.replace("-", "_")
        script_flow_dir = Path(os.path.join(
            os.getcwd(), "scripts", "promptflow-ci", "test_configs", "flows", flow_dir_name))

        # Call MT to submit flow
        # Skip template flow
        if (section_type == 'template'):
            log_debug(f"Skipped template flow: {flow_dir}. Flow id: {flow_id}")
            continue
        sample_path = flow_dir / "samples.json"
        log_debug(f"Sample input file path: {sample_path}")
        if not sample_path.exists():
            raise Exception(
                f"Sample input file path doesn't exist when submitting flow {flow_dir}")
        batch_data_inputs = _resolve_data_to_asset_id(sample_path)
        log_debug(
            f"\nStarting to submit bulk test run. Flow dir: {flow_dir_name}. Flow id: {flow_id}")
        submit_flow_payload = flow_utils.construct_submit_flow_payload_of_new_contract(
            flow_id, batch_data_inputs, runtime_name, flow_dag, args.flow_submit_mode
        )

        experiment_id = flow_create_result["experimentId"]
        try:
            submit_flow_result, flow_run_id, _ = mt_client.submit_flow(
                submit_flow_payload, experiment_id)
            bulk_test_run_id = submit_flow_result["bulkTestId"]
            flow_run_ids = flow_utils.get_flow_run_ids(submit_flow_result)
        except Exception as e:
            failure_message = f"Submit bulk test run failed. Flow dir: {flow_dir}. Flow id: {flow_id}. Error: {e}"
            log_error(failure_message)
            handled_failures.append(failure_message)
        else:
            debug_output(submit_flow_result, "submit_flow_result",
                         flow_dir.name, args.local)
            log_debug(
                f"All the flow run links for bulk test: {bulk_test_run_id}")
            for run_id in flow_run_ids:
                submitted_flow_run_identifiers.add(
                    flow_utils.create_flow_run_identifier(flow_id, run_id))
                flow_run_link = flow_utils.get_flow_run_link(submit_flow_result, ux_endpoint,
                                                             args.subscription_id, args.resource_group,
                                                             args.workspace_name, experiment_id, run_id)
                log_debug(
                    f"Flow run link for run {run_id} to Azure Machine Learning Portal: {flow_run_link}")

            # Do bulk test for evaluation flow
        if flow_type == "evaluate":
            if not os.path.exists(script_flow_dir / "bulk_test_config_of_new_contract.json"):
                raise Exception(
                    f"bulk_test_config_of_new_contract.json not found in {script_flow_dir}. Please provide "
                    f"bulk_test_config_of_new_contract.json to the evaluation flow {flow_dir.name}.")
            elif not os.path.exists(script_flow_dir / "samples.json"):
                raise Exception(f"Sample input file path doesn't exist when doing main flow + evaluation flow bulk "
                                f"test. Flow dir: {flow_dir.name}.")
            else:
                log_debug(
                    f"\nStarting to do main flow + evaluation flow bulk test.\nEvaluation flow dir: {flow_dir_name}. "
                    f"Evaluation flow id: {flow_id}. Flow resource id: {flow_resource_id}")
                try:
                    main_flow_folder_name, bulk_test_config = flow_utils.resolve_bulk_test_config_of_new_contract(
                        script_flow_dir
                    )
                    if main_flow_folder_name in bulk_test_main_flows_creation_info:
                        main_flow_creation_info = bulk_test_main_flows_creation_info[
                            main_flow_folder_name]
                    else:
                        raise Exception(
                            f"Sample flow {main_flow_folder_name} is not found in the flows folder. Please make sure "
                            f"the sample flow is provided.")
                    main_flow_id = main_flow_creation_info[1]['flowId']
                    main_flow_dag = main_flow_creation_info[0]
                    main_flow_name = main_flow_creation_info[1]['flowName']
                    log_debug(
                        f"Main flow dir: {main_flow_folder_name}. Main flow id: {main_flow_id}.")

                    batch_data_inputs = _resolve_data_to_asset_id(
                        flow_dir / "samples.json")
                    submit_bulk_test_payload, evaluation_run_id = \
                        flow_utils.construct_submit_bulk_test_payload_of_new_contract(
                            evaluation_flow_dag=flow_dag,
                            evaluation_flow_resource_id=flow_resource_id,
                            batch_data_inputs=batch_data_inputs,
                            bulk_test_config=bulk_test_config,
                            main_flow_id=main_flow_id,
                            main_flow_dag=main_flow_dag,
                            runtime_name=runtime_name,
                            evaluation_flow_name=flow_name,
                            main_flow_name=main_flow_name,
                        )
                    debug_output(
                        submit_bulk_test_payload, "submit_bulk_test_payload", flow_dir.name, args.local)
                except Exception as e:
                    log_error(
                        f"Error when constructing evaluation bulk test payload for {flow_dir_name}")
                    raise e
                else:
                    # Call MT to submit bulk test & evaluation
                    try:
                        submit_bulk_test_result, _, evaluation_run_id = mt_client.submit_flow(
                            submit_bulk_test_payload,
                            experiment_id
                        )
                        bulk_test_run_id = submit_flow_result["bulkTestId"]
                        flow_run_ids = flow_utils.get_flow_run_ids(
                            submit_bulk_test_result)
                    except Exception as e:
                        failure_message = f"Submit bulk test evaluation run failed. Flow dir: {flow_dir}. Evaluation " \
                                          f"flow id: {flow_id}. Main flow id: {main_flow_id}. Error: {e}"
                        log_error(failure_message)
                        handled_failures.append(failure_message)
                    else:
                        debug_output(submit_bulk_test_result, "submit_bulk_test_response", flow_dir.name,
                                     args.local)
                        log_debug(
                            f"All the flow run links for bulk test: {bulk_test_run_id}")
                        for run_id in flow_run_ids:
                            submitted_bulk_test_run_identifiers.add(
                                flow_utils.create_flow_run_identifier(
                                    main_flow_id, run_id)
                            )
                            flow_run_link = flow_utils.get_flow_run_link(submit_bulk_test_result,
                                                                         ux_endpoint,
                                                                         args.subscription_id,
                                                                         args.resource_group,
                                                                         args.workspace_name,
                                                                         experiment_id,
                                                                         run_id
                                                                         )
                            log_debug(f"Flow run link for run {run_id} to Azure Machine Learning Portal: "
                                      f"{flow_run_link}")
    # endregion

    # region: Step3. check the submitted run status
    check_run_status_interval = 30  # seconds
    check_run_status_max_attempts = 30  # times
    flow_runs_count = len(submitted_flow_run_identifiers)
    bulk_test_runs_count = len(submitted_bulk_test_run_identifiers)
    failed_flow_runs = {}  # run key : flow_run_link
    failed_evaluation_runs = {}  # run key : evaluation_run_link
    if flow_runs_count == 0 and bulk_test_runs_count == 0:
        log_debug(
            "\nNo bulk test run or bulk test evaluation run need to check status")

    run_workspace = Workspace.get(
        name=run_operations._operation_scope.workspace_name,
        subscription_id=run_operations._operation_scope.subscription_id,
        resource_group=run_operations._operation_scope.resource_group_name,
    )

    flow_runs_to_check = copy.deepcopy(submitted_flow_run_identifiers)
    log_debug(f"\n{flow_runs_count} bulk test runs need to check status.")
    check_flow_run_status(flow_runs_to_check, submitted_flow_run_identifiers, check_run_status_interval, check_run_status_max_attempts)

    bulk_test_runs_to_check = copy.deepcopy(
        submitted_bulk_test_run_identifiers)
    log_debug(
        f"\n{bulk_test_runs_count} bulk test evaluation runs need to check status.")
    check_flow_run_status(bulk_test_runs_to_check,
                          submitted_bulk_test_run_identifiers, check_run_status_interval, check_run_status_max_attempts)

    if len(submitted_flow_run_identifiers) > 0 or len(submitted_bulk_test_run_identifiers) > 0:
        failure_message = f"Not all bulk test runs or bulk test evaluation runs are completed after " \
                          f"{check_run_status_max_attempts} attempts. " \
                          f"Please check the run status on Azure Machine Learning Portal."
        log_error(failure_message, True)
        if args.flow_submit_mode == "async":
            log_warning(
                "Not fail the CI because the flow submission mode is async.")
        else:
            handled_failures.append(failure_message)

        for flow_run_identifier in submitted_flow_run_identifiers:
            flow_id, flow_run_id = flow_utils.resolve_flow_run_identifier(
                flow_run_identifier)
            flow_run_link = flow_utils.construct_flow_run_link(ux_endpoint, args.subscription_id,
                                                               args.resource_group, args.workspace_name,
                                                               experiment_id, flow_id, flow_run_id)
            log_debug(
                f"Flow run link for run {flow_run_id} to Azure Machine Learning Portal: {flow_run_link}")
        for bulk_test_run_identifier in submitted_bulk_test_run_identifiers:
            flow_id, flow_run_id = flow_utils.resolve_flow_run_identifier(
                bulk_test_run_identifier)
            flow_run_link = flow_utils.construct_flow_run_link(ux_endpoint, args.subscription_id,
                                                               args.resource_group, args.workspace_name,
                                                               experiment_id, flow_id, flow_run_id)
            log_debug(
                f"Flow run link for run {flow_run_id} to Azure Machine Learning Portal: {flow_run_link}")

    if len(failed_flow_runs) > 0 or len(failed_evaluation_runs) > 0:
        failure_message = "There are bulk test runs or bulk test evaluation runs failed. " \
                          "Please check the run error on Azure Machine Learning Portal."
        log_error(failure_message, True)
        handled_failures.append(failure_message)
        for flow_run_key, flow_run_link in failed_flow_runs.items():
            log_error(
                f"Bulk test run link to Azure Machine Learning Portal: {flow_run_link}")
        for flow_run_key, evaluation_run_link in failed_evaluation_runs.items():
            log_error(
                f"Bulk test evaluation run link to Azure Machine Learning Portal: {evaluation_run_link}")
    elif len(submitted_flow_run_identifiers) == 0 and len(submitted_bulk_test_run_identifiers) == 0:
        log_debug(f"\nRun status checking completed. {flow_runs_count} flow runs and {bulk_test_runs_count} bulk "
                  f"test evaluation runs completed.")
    # Fail CI if there are failures.
    if len(handled_failures) > 0:
        log_error(
            "Promptflow CI failed due to the following failures, please check the error logs for details.", True)
        for failure in handled_failures:
            log_error(failure)
        exit(1)
    # endregion
