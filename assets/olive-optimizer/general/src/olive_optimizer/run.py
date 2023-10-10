# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""This file contains the core logic for aml olive optimizer component."""

import os
import sys
import json
import time
import argparse
import logging
import shutil
import subprocess
from pathlib import Path
from zipfile import ZipFile
from azureml.core import Run
from olive.workflows import run as olive_run

logging.basicConfig(stream=sys.stdout,
                    level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')
log = logging.getLogger("online_endpoints_model_optimizer.optimize")

# output env vars
output_root_path = os.getenv("AZUREML_CR_EXECUTION_WORKING_DIR_PATH", os.getenv("AZ_BATCHAI_JOB_WORK_DIR", "./"))
output_path = f"{output_root_path}/outputs"
workdir = "/tmp/workdir"
os.makedirs(workdir, exist_ok=False)


def _parse_arguments():
    """Parse the arguments passed to the script, check the validity of the arguments."""
    parser = argparse.ArgumentParser(description="Online Endpoints Model Optimizer",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config_path",
        default=None,
        type=str,
        required=True,
        help="Input: Path to the configuration file"
    )
    parser.add_argument(
        "--model_path",
        default=None,
        type=str,
        required=False,
        help="Input: Path to the model folder"
    )
    parser.add_argument(
        "--code",
        default=None,
        type=str,
        required=False,
        help="Input: Path to the code directory, including user script, user script dependencies, requirements etc"
    )
    parser.add_argument(
        "--data_path",
        default=None,
        type=str,
        required=False,
        help="Input: Path to the data directory, including data directory such as data_dir, train_data_dir etc"
    )
    parser.add_argument(
        "--optimized_parameters_path",
        default=None,
        type=str,
        required=False,
        help="OutputPath: Path to the inference parameters configs for optimized models"
    )
    parser.add_argument(
        "--optimized_model_path",
        default=None,
        type=str,
        required=False,
        help="OutputPath: Path to the optimized model"
    )

    arguments = parser.parse_args()
    config_path = arguments.config_path
    model_folder_path = arguments.model_path
    code_folder_path = arguments.code
    data_folder_path = arguments.data_path
    optimized_parameters_path = arguments.optimized_parameters_path
    optimized_model_path = arguments.optimized_model_path

    log.info(f"config_path: {config_path}")
    log.info(f"model_folder_path: {model_folder_path}")
    log.info(f"code_folder_path: {code_folder_path}")
    log.info(f"data_folder_path: {data_folder_path}")
    log.info(f"optimized_parameters_path: {optimized_parameters_path}")
    log.info(f"optimized_model_path: {optimized_model_path}")

    if os.path.isfile(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
    else:
        config = json.loads(config_path)

    _validate_config(config)

    if model_folder_path is not None:
        shutil.copytree(model_folder_path, workdir, dirs_exist_ok=True)
        log.info(f"replace all files in model folder {model_folder_path} to {workdir}")

    if code_folder_path is not None:
        shutil.copytree(code_folder_path, workdir, dirs_exist_ok=True)
        log.info(f"move all files in code folder {code_folder_path} to {workdir}")

    _set_default_setting(config)
    log.info(f"config: {config}")
    return config, code_folder_path, data_folder_path, optimized_parameters_path, optimized_model_path


def _validate_config(config):
    """Validate the config file."""
    if 'systems' in config.keys() and ("aml_system" in config['systems'] or "docker_system" in config['systems']):
        raise Exception("aml_system or docker_system is not supported yet, will exit!")
    if "engine" not in config.keys():
        raise Exception("engine is not found in config, will exit!")


def _set_default_setting(config):
    """Set default setting for the config file."""
    log.info(f"set output_dir to: {output_path}")
    config['engine']['output_dir'] = output_path
    if "output_model_num" not in config['engine']['search_strategy'].keys():
        config['engine']['search_strategy']['output_model_num'] = 1
        log.info("set output_model_num to: 1")
    if "plot_pareto_frontier" not in config['engine'].keys():
        config['engine']['plot_pareto_frontier'] = True
        log.info("set plot_pareto_frontier to: true")
    config['engine']['packaging_config'] = {}


def _move_model_and_config_to_output_path(optimized_parameters_path, optimized_model_path):
    """Move the optimized model and config to output path."""
    os.chdir(output_path)

    with ZipFile("OutputModels.zip", mode="r") as zip_ref:
        zip_ref.extractall("OutputModels")

    for dirpath, dirs, files in os.walk("OutputModels/CandidateModels"):
        for file in files:
            parentpath = os.path.split(dirpath)
            prefix = os.path.split(parentpath[0])[1] + "_" + parentpath[1]
            if file.endswith("inference_config.json") and optimized_parameters_path is not None:
                log.info(f"copy inference_config {os.path.join(dirpath, file)} to "
                         f"{optimized_parameters_path}/{prefix}_inference_config.json")
                shutil.copy2(os.path.join(dirpath, file),
                             f"{optimized_parameters_path}/{prefix}_inference_config.json")
            elif file.endswith("metrics.json"):
                with open(os.path.join(dirpath, file), "r") as f:
                    metrics = json.load(f)
                    _report_job_metrics(f"metrics_value_{prefix}", metrics)
            elif file.endswith("model.onnx") and optimized_model_path is not None:
                log.info(f"copy optimized model {os.path.join(dirpath, file)} to "
                         f"{optimized_model_path}/{prefix}_model/model.onnx")
                os.makedirs(f"{optimized_model_path}/{prefix}_model", exist_ok=True)
                shutil.copy2(os.path.join(dirpath, file),
                             f"{optimized_model_path}/{prefix}_model/model.onnx")
            elif file.endswith("model.onnx.data") and optimized_model_path is not None:
                log.info(f"copy optimized model data {os.path.join(dirpath, file)} to "
                         f"{optimized_model_path}/{prefix}/model.onnx.data")
                os.makedirs(f"{optimized_model_path}/{prefix}_model", exist_ok=True)
                shutil.copy2(os.path.join(dirpath, file),
                             f"{optimized_model_path}/{prefix}_model/model.onnx.data")
            elif dirpath.endswith("model") and optimized_model_path is not None:
                openvinopath = os.path.split(Path(Path(dirpath).parent))
                prefix = os.path.split(openvinopath[0])[1] + "_" + openvinopath[1]
                log.info(f"copy optimized model {dirpath} to {optimized_model_path}/{prefix}_model")
                shutil.copytree(dirpath, f"{optimized_model_path}/{prefix}_model", dirs_exist_ok=True)


def _report_job_metrics(name, values):
    """Report job metrics."""
    job_metrics = {}
    for param in values:
        log.info(f"Report job metrics for {param}")
        value = str(values[param])
        Run.get_context().log(name=param, value=value)
        job_metrics[param] = [value]

    Run.get_context().log_table(name=name, value=job_metrics)
    time.sleep(2)


def run():
    """Invoke the olive_run."""
    config, code_folder_path, data_folder_path, optimized_parameters_path, optimized_model_path = _parse_arguments()
    os.chdir(workdir)
    if code_folder_path is not None:
        log.info("Current working directory: {0}".format(os.getcwd()))
        if os.path.exists("requirements.txt"):
            log.info("pip install requirements")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    try:
        if data_folder_path is not None:
            olive_run(config=config, data_root=data_folder_path)
        else:
            olive_run(config)
        _move_model_and_config_to_output_path(optimized_parameters_path, optimized_model_path)
    except Exception as exception:
        log.exception(exception)
        raise


if __name__ == '__main__':
    run()
