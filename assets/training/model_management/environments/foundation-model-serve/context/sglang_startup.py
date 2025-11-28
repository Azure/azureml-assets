import os
import sys
import subprocess

"""Initialize client[s] for the SGLang engine to receive requests on."""

# Step 0: Detect number of GPUs
def get_gpu_count():
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        # Fallback: try nvidia-smi command
        try:
            output = subprocess.check_output("nvidia-smi -L", shell=True, stderr=subprocess.STDOUT)
            return len(output.decode().strip().splitlines())
        except Exception:
            # Default to 1 if GPU detection fails
            return 1

gpu_count = get_gpu_count()
os.environ["AML_TENSOR_PARALLEL_SIZE"] = str(gpu_count)

# Read environment variables
azureml_model_dir = os.getenv("AZUREML_MODEL_DIR")
aml_model = os.getenv("AML_MODEL_PATH")
os.environ["AML_PORT"] = os.getenv("ENGINE_STARTUP_PORT", "8000")

final_model_path = ""

# Step 1: Compose final model path
if azureml_model_dir:
    final_model_path = azureml_model_dir

if aml_model:
    # Strip leading slashes to prevent absolute path override
    aml_model = aml_model.lstrip("/")
    final_model_path = os.path.join(final_model_path, aml_model) if final_model_path else aml_model

if final_model_path:
    os.environ["AML_MODEL"] = final_model_path

# Step 2: Build SGLang command
prefix = "AML_"
cmd = [sys.executable, "-m", "sglang.launch_server"]

# Map all environment variables starting with "AML_" into command-line args
for key, value in os.environ.items():
    if key.startswith(prefix):
        arg_name = key[len(prefix):].lower().replace("_", "-")
        cmd.extend([f"--{arg_name}", value])

# Step 3: Print and start process
print("Starting SGLang server with command:")
print(" ".join(cmd))

subprocess.Popen(cmd)
