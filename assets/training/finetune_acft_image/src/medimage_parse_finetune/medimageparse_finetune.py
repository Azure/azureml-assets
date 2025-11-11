# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Script to execute olympus_core training with command line arguments."""

import argparse
import os
import sys
import shutil
from pathlib import Path
from azureml.acft.image.components.olympus_biomed_parse.checkpoint_loaders.safetensors_loader import (
    convert_ckpt_to_safetensor,
)
from azureml.dataprep.api._loggerfactory import _LoggerFactory
import importlib.resources
import subprocess
import contextlib
import signal
import tempfile

logger = _LoggerFactory.get_logger(__name__)

# Discover ranks from AML env (AML sets these for PyTorch distribution)
RANK = int(os.environ.get("AZUREML_CR_NODE_RANK", "0"))  # machine index: 0,1,...
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))  # proc index on a node
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
print(f"[launcher] NODE_RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Execute MedImageParse fine-tuning")

    parser.add_argument(
        "--pretrained_mlflow_model",
        type=str,
        required=True,
        help="Path to the MLflow model from azureml registry. This is the Model that will be fine-tuned.",
    )

    parser.add_argument(
        "--data", type=str, required=True, help="Path to the finetune data directory"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the parameters.yaml config file",
    )

    parser.add_argument(
        "--out", type=str, required=True, help="Path to the output directory"
    )

    parser.add_argument(
        "--mlflow_model_folder",
        type=str,
        required=True,
        help="Path to save the output model after fine-tuning",
    )

    return parser.parse_args()


def is_mpi_enabled():
    """Check if MPI is enabled in the environment."""
    return "OMPI_COMM_WORLD_RANK" in os.environ


def safe_setenv(var, value):
    """Set environment variable only if not already set to a different value."""
    if value is None:
        return False
    if var in os.environ and os.environ[var] is not None and os.environ[var] != value:
        logger.info(f"Env '{var}' already set to '{os.environ[var]}', not changing to '{value}'")
        return False
    os.environ[var] = value
    return True


def set_environment_variables_for_nccl_backend(master_port=6105):
    """Set environment variables for NCCL backend."""
    if is_mpi_enabled():
        safe_setenv("RANK", os.environ.get("OMPI_COMM_WORLD_RANK"))
        safe_setenv("WORLD_SIZE", os.environ.get("OMPI_COMM_WORLD_SIZE"))
        safe_setenv("LOCAL_RANK", os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK"))
        safe_setenv("MASTER_ADDR", os.environ.get("MASTER_ADDR") or "127.0.0.1")
        safe_setenv("MASTER_PORT", os.environ.get("MASTER_PORT") or str(master_port))


def _is_heavy_model_file(rel_path: Path) -> bool:
    """Return True if the suffix is .safetensors."""
    _HEAVY_EXTS = {".safetensors"}

    name = rel_path.name
    if name == "model.safetensors.index.json":
        return False

    if rel_path.suffix in _HEAVY_EXTS:
        return True

    return False


def copy_catalog_except_weights(src_model_dir: str, dst_model_dir: str):
    """Recursively copy the MLflow model directory from src to dst, skipping .safetensors."""
    src = Path(src_model_dir)
    dst = Path(dst_model_dir)

    logger.info("=== STARTING COPY CATALOG ===")
    logger.info("Copying catalog model (excluding heavy weights):")
    logger.info(f"  src = {src}")
    logger.info(f"  dst = {dst}")
    logger.info(f"  src exists = {src.exists()}")

    if not src.exists():
        logger.error(f"Source model dir does not exist: {src}")
        return

    logger.info("Source directory exists, starting file walk...")

    file_count = 0
    for root, dirs, files in os.walk(src):
        logger.info(f"Walking directory: {root}")
        logger.info(f"  Found {len(files)} files, {len(dirs)} directories")

        rel_root = Path(root).relative_to(src)
        target_root = dst / rel_root
        target_root.mkdir(parents=True, exist_ok=True)

        for fname in files:
            file_count += 1
            rel_file = rel_root / fname
            logger.info(f"Processing file {file_count}: {rel_file}")

            if _is_heavy_model_file(rel_file):
                logger.info(f"  skip heavy: {rel_file}")
                continue

            src_file = src / rel_file
            dst_file = dst / rel_file
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
            logger.info(f"  copied: {rel_file}")

    logger.info(f"Catalog copy complete. Processed {file_count} files total.")
    logger.info("=== COPY CATALOG FINISHED ===")


@contextlib.contextmanager
def with_signal_handler(process):
    """Context manager to forward SIGTERM and SIGINT to a subprocess."""
    def handler(sig, frame):
        process.send_signal(sig)
    orig_sigterm = signal.signal(signal.SIGTERM, handler)
    orig_sigint = signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, orig_sigterm)
        signal.signal(signal.SIGINT, orig_sigint)


def execute_training(args):
    """Execute the olympus_core training command directly."""
    logger.info("Received arguments:")
    logger.info(f"  --pretrained_mlflow_model: {args.pretrained_mlflow_model}")
    logger.info(f"  --data: {args.data}")
    logger.info(f"  --config: {args.config}")
    logger.info(f"  --out: {args.out}")
    logger.info(f"  --mlflow_model_folder: {args.mlflow_model_folder}")

    # Set up environment variables
    os.environ["OUTPUT"] = args.out
    os.environ["EXTERNAL"] = args.data
    os.environ["AMLT_EXPERIMENT_NAME"] = "MIP_finetune"
    os.environ["AMLT_JOB_NAME"] = "MIP_finetune_job"
    os.environ["MODEL_PATH"] = args.pretrained_mlflow_model
    os.environ["FINETUNED_MODEL_PATH"] = args.mlflow_model_folder

    # Set up distributed training environment variables
    set_environment_variables_for_nccl_backend()

    logger.info("Environment variables set:")
    logger.info(f"  OUTPUT={os.environ['OUTPUT']}")
    logger.info(f"  EXTERNAL={os.environ['EXTERNAL']}")
    logger.info(f"  AMLT_EXPERIMENT_NAME={os.environ['AMLT_EXPERIMENT_NAME']}")
    logger.info(f"  AMLT_JOB_NAME={os.environ['AMLT_JOB_NAME']}")
    logger.info(f"  MODEL_PATH={os.environ['MODEL_PATH']}")
    logger.info(f"  FINETUNED_MODEL_PATH={os.environ['FINETUNED_MODEL_PATH']}")

    # Add debug logging before calling the function
    logger.info(
        f"About to copy catalog from {args.pretrained_mlflow_model} to {args.mlflow_model_folder}"
    )
    logger.info(f"Source exists: {Path(args.pretrained_mlflow_model).exists()}")
    logger.info(f"Destination exists: {Path(args.mlflow_model_folder).exists()}")

    # IMPORTANT: point OUTPUT env to node-local temp for non-rank0
    node_local_output = str(Path(tempfile.gettempdir()) / f"mip_train_node{RANK}")
    Path(node_local_output).mkdir(parents=True, exist_ok=True)
    if RANK == 0:
        os.environ["OUTPUT"] = args.out
    else:
        os.environ["OUTPUT"] = node_local_output

    # Always safe to set EXTERNAL and MLFLOW_MODEL_FOLDER
    os.environ["EXTERNAL"] = args.data
    os.environ["MLFLOW_MODEL_FOLDER"] = args.pretrained_mlflow_model

    # ONLY node 0 may touch AzureML outputs
    if RANK == 0:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        Path(args.mlflow_model_folder).mkdir(parents=True, exist_ok=True)
        copy_catalog_except_weights(args.pretrained_mlflow_model, args.mlflow_model_folder)

    logger.info("Copy catalog function completed")

    # Use static shell script
    script_path = Path(__file__).parent
    launcher_script = script_path / "train_launcher.sh"

    # Check if launcher script exists
    if not launcher_script.exists():
        logger.error(f"Launcher script not found: {launcher_script}")
        return 1

    # Prepare arguments for olympus_core
    olympus_args = [
        "-cp",
        str(
            importlib.resources.files(
                "azureml.acft.image.components.olympus_biomed_parse"
            ) / "configs"
        ),
        "-cn", os.environ["AMLT_EXPERIMENT_NAME"],
        "-cd", str(Path(args.config).parent),
    ]

    logger.info("Calling olympus_core with arguments:")
    logger.info(" ".join(olympus_args))
    cmd = " ".join([str(launcher_script)] + olympus_args)
    proc = subprocess.Popen(cmd,
                            shell=True,
                            executable=shutil.which("bash") or None,
                            stdout=sys.stdout,
                            stderr=sys.stderr
                            )

    with with_signal_handler(proc):
        exit_code = proc.wait()

    if exit_code == 0:
        logger.info("Training completed successfully.")

        if RANK == 0:
            ckpt_path = (
                Path(args.out)
                / os.environ['AMLT_EXPERIMENT_NAME']
                / os.environ['AMLT_JOB_NAME']
                / "checkpoints"
                / "last.ckpt"
            )
            output_path = (
                Path(args.mlflow_model_folder)
                / "artifacts"
                / "checkpoints"
                / "boltzformer_focal_all.safetensors"
            )

            logger.info("Starting conversion to HuggingFace format...")
            convert_ckpt_to_safetensor(str(ckpt_path), str(output_path))

            logger.info("Output directory snapshot:")
            for item in Path(args.out).rglob("*"):
                if item.is_file():
                    logger.info(f"  {item}")

            logger.info("Finetuned MLflow model snapshot:")
            for item in Path(args.mlflow_model_folder).rglob("*"):
                if item.is_file():
                    logger.info(f"  {item}")
    else:
        logger.error(f"Training failed with exit code: {exit_code}")

    return exit_code


def main():
    """Run the MedImageParse fine-tuning script."""
    logger.info("MedImageParse Fine-tuning Script")
    logger.info("=" * 40)

    # Parse arguments
    args = parse_arguments()

    # Validate paths exist
    paths_to_check = [
        ("data", args.data),
        ("mlflow_model_folder", args.mlflow_model_folder),
    ]

    # For config, check if it exists (can be file or directory)
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config path does not exist: {args.config}")
        return 1

    for name, path_str in paths_to_check:
        path = Path(path_str)
        if not path.exists():
            logger.error(f"{name} path does not exist: {path}")
            return 1
        if not path.is_dir():
            logger.error(f"{name} path is not a directory: {path}")
            return 1

    # Create output directory if it doesn't exist
    if RANK == 0:
        out_path = Path(args.out)
        if not out_path.exists():
            logger.info(f"Creating output directory: {out_path}")
            out_path.mkdir(parents=True, exist_ok=True)
        
        checkpoint_path = Path(args.mlflow_model_folder) / "artifacts" / "checkpoints"
        if not checkpoint_path.exists():
            logger.info(f"Creating output directory: {checkpoint_path}")
            checkpoint_path.mkdir(parents=True, exist_ok=True)

    # Execute training
    return execute_training(args)


if __name__ == "__main__":
    sys.exit(main())
