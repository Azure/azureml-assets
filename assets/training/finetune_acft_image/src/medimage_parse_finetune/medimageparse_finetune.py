# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Script to execute olympus_core training with command line arguments
"""

import argparse
import os
import sys
import shutil
from pathlib import Path
from azureml.acft.image.components.olympus.app.main import main as olympus_main
from azureml.acft.image.components.olympus_biomed_parse.checkpoint_loaders.safetensors_loader import convert_ckpt_to_safetensor
from azureml.dataprep.api._loggerfactory import _LoggerFactory
import importlib.resources

logger = _LoggerFactory.get_logger(__name__)

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Execute MedImageParse fine-tuning")
    
    parser.add_argument(
        "--pretrained_mlflow_model",
        type=str,
        required=True,
        help="Path to the MLflow model from azureml registry. This is the Model that will be fine-tuned."
    )
    
    parser.add_argument(
        "--data",
        type=str,
        required=True,
        help="Path to the finetune data directory"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the parameters.yaml config file"
    )
    
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Path to the output directory"
    )

    parser.add_argument(
        "--mlflow_model_folder",
        type=str,
        required=True,
        help="Path to save the output model after fine-tuning"
    )
    
    return parser.parse_args()


def _is_heavy_model_file(rel_path: Path) -> bool:
    """
    Return True if the suffix is .safetensors (heavy file). Return False otherwise
    We always KEEP the index JSON (model.safetensors.index.json) even though it sits next to weight shards.
    """
    _HEAVY_EXTS = {".safetensors"}

    name = rel_path.name
    if name == "model.safetensors.index.json":
        return False

    if rel_path.suffix in _HEAVY_EXTS:
        return True

    return False


def copy_catalog_except_weights(src_model_dir: str, dst_model_dir: str):
    """
    Recursively copy the MLflow model directory from src to dst, skipping heavy
    weight files by extension (safetensors).
    """
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


def execute_training(args):
    """Execute the olympus_core training command directly"""
    
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

    logger.info("Environment variables set:")
    logger.info(f"  OUTPUT={os.environ['OUTPUT']}")
    logger.info(f"  EXTERNAL={os.environ['EXTERNAL']}")
    logger.info(f"  AMLT_EXPERIMENT_NAME={os.environ['AMLT_EXPERIMENT_NAME']}")
    logger.info(f"  AMLT_JOB_NAME={os.environ['AMLT_JOB_NAME']}")
    logger.info(f"  MODEL_PATH={os.environ['MODEL_PATH']}")
    logger.info(f"  FINETUNED_MODEL_PATH={os.environ['FINETUNED_MODEL_PATH']}")

    # Add debug logging before calling the function
    logger.info(f"About to copy catalog from {args.pretrained_mlflow_model} to {args.mlflow_model_folder}")
    logger.info(f"Source exists: {os.path.exists(args.pretrained_mlflow_model)}")
    logger.info(f"Destination exists: {os.path.exists(args.mlflow_model_folder)}")

    copy_catalog_except_weights(args.pretrained_mlflow_model, args.mlflow_model_folder)

    logger.info("Copy catalog function completed")
    
    # Prepare arguments for olympus_core
    olympus_args = [
        "-cp", str(importlib.resources.files('azureml.acft.image.components.olympus_biomed_parse') / "configs"),
        "-cn", os.environ['AMLT_EXPERIMENT_NAME'],
        "-cd", str(Path(args.config).parent)
    ]
    
    logger.info("Calling olympus_core with arguments:")
    logger.info(" ".join(olympus_args))
    
    # Save original sys.argv
    original_argv = sys.argv.copy()
    
    try:        
        # Set sys.argv for olympus_core
        sys.argv = ["olympus_main"] + olympus_args
        
        # Call olympus_core main function directly
        result = olympus_main()

        ckpt_path = args.out + f"/{os.environ['AMLT_EXPERIMENT_NAME']}/{os.environ['AMLT_JOB_NAME']}/checkpoints/last.ckpt"
        output_path = args.mlflow_model_folder + "/artifacts/checkpoints/boltzformer_focal_all.safetensors"

        convert_ckpt_to_safetensor(ckpt_path, output_path)
        
        logger.info("Olympus training completed successfully")
        return result if result is not None else 0
        
    except SystemExit as e:
        # Handle sys.exit calls from olympus_core
        logger.info(f"Olympus training exited with code: {e.code}")
        return e.code if e.code is not None else 0
        
    except Exception as e:
        logger.error(f"Error during olympus training: {e}")
        logger.error(f"Error type: {type(e).__name__}")
        return 1

    finally:
        # Restore original sys.argv
        sys.argv = original_argv


def main():
    """Main function"""
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
    if not os.path.exists(args.config):
        logger.error(f"Config path does not exist: {args.config}")
        return 1

    for name, path in paths_to_check:
        if not os.path.exists(path):
            logger.error(f"{name} path does not exist: {path}")
            return 1
        if not os.path.isdir(path):
            logger.error(f"{name} path is not a directory: {path}")
            return 1

    # Create output directory if it doesn't exist
    if not os.path.exists(args.out):
        logger.info(f"Creating output directory: {args.out}")
        os.makedirs(args.out, exist_ok=True)
    elif not os.path.exists(args.mlflow_model_folder + '/artifacts/checkpoints'):
        logger.info(f"Creating output directory: {args.mlflow_model_folder + '/artifacts/checkpoints'}")
        os.makedirs(args.mlflow_model_folder + '/artifacts/checkpoints', exist_ok=True)

    # Execute training
    return execute_training(args)


if __name__ == "__main__":
    sys.exit(main())
