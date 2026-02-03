# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""VLLM Engine module.

This module contains the VLLMEngine class which is responsible for initializing the VLLM server,
generating responses for given prompts, and managing the server processes.
"""

import subprocess
import sys
import os
from foundation.model.serve.constants import EnvironmentVariables, CommonConstants
from foundation.model.serve.engine.engine import BaseEngine
from foundation.model.serve.logging_config import configure_logger

logger = configure_logger(__name__)


class VLLMEngine(BaseEngine):
    """VLLM Engine class.

    This class is responsible for initializing the VLLM server, generating responses for given prompts,
    and managing the server processes.
    """

    def __init__(self):
        """Initialize the VLLMEngine with the given engine and task configurations."""
        pass

    def init_server(self):
        """Initialize client[s] for the engine to receive requests on.

        Starts the VLLM server process with appropriate command-line arguments
        derived from environment variables.
        """
        self.formulate_environment_variables()

        prefix = "AML_"
        cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]

        for key, value in os.environ.items():
            if key.startswith(prefix):
                arg_name = key[len(prefix):].lower().replace("_", "-")
                cmd.extend([f"--{arg_name}", value])

        print("Starting vLLM server with command:")
        print(" ".join(cmd))

        subprocess.Popen(cmd)

    def formulate_environment_variables(self):
        """Formulate environment variables specific to VLLM engine.

        Processes AML-specific environment variables and sets up the model path
        and port configuration for the VLLM engine.

        Returns:
            str: The final model path to be used by VLLM.

        Raises:
            EnvironmentError: If required environment variables are not set.
        """
        azureml_model_dir = os.getenv(EnvironmentVariables.AZUREML_MODEL_DIR)
        aml_model = os.getenv(EnvironmentVariables.AML_MODEL_PATH)
        engine_port = os.getenv(
            EnvironmentVariables.ENGINE_STARTUP_PORT, str(CommonConstants.DEFAULT_PORT))

        if not aml_model:
            raise EnvironmentError(
                f"{EnvironmentVariables.AML_MODEL} environment variable is not set.")

        # Set AML port
        os.environ[EnvironmentVariables.AML_PORT] = engine_port

        # Normalize and construct model path
        aml_model = aml_model.lstrip("/")
        final_model_path = (
            os.path.join(azureml_model_dir, aml_model)
            if azureml_model_dir else aml_model
        )

        os.environ[EnvironmentVariables.AML_MODEL] = final_model_path

        # Set tensor parallel size only if not already set
        if not os.getenv(EnvironmentVariables.AML_TENSOR_PARALLEL_SIZE):
            # Try using torch first
            try:
                import torch
                gpu_count = torch.cuda.device_count()
            except Exception:
                # Fallback: try using nvidia-smi
                try:
                    gpu_info = subprocess.check_output(["nvidia-smi", "-L"]).decode()
                    gpu_count = gpu_info.count("UUID")
                except Exception:
                    gpu_count = 0  # No GPU detected
            os.environ[EnvironmentVariables.AML_TENSOR_PARALLEL_SIZE] = str(gpu_count)
            logger.info(f"Set {EnvironmentVariables.AML_TENSOR_PARALLEL_SIZE} to {gpu_count}")
        else:
            logger.info(f"{EnvironmentVariables.AML_TENSOR_PARALLEL_SIZE} already set to "
                        f"{os.getenv(EnvironmentVariables.AML_TENSOR_PARALLEL_SIZE)}")
