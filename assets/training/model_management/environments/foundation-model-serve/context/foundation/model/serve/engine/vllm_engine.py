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
        """Initialize client[s] for the engine to receive requests on."""
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
        """Formulate environment variables specific to VLLM engine."""
        azureml_model_dir = os.getenv(EnvironmentVariables.AZUREML_MODEL_DIR)
        aml_model = os.getenv(EnvironmentVariables.AML_MODEL)
        engine_port = os.getenv(EnvironmentVariables.ENGINE_STARTUP_PORT, str(CommonConstants.DEFAULT_PORT))

        if not aml_model:
            raise EnvironmentError(f"{EnvironmentVariables.AML_MODEL} environment variable is not set.")

        # Set AML port
        os.environ[EnvironmentVariables.AML_PORT] = engine_port

        # Normalize and construct model path
        aml_model = aml_model.lstrip("/")
        final_model_path = (
            os.path.join(azureml_model_dir, aml_model)
            if azureml_model_dir else aml_model
        )

        os.environ[EnvironmentVariables.AML_MODEL] = final_model_path

        return final_model_path
