# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Script to list checkpoint directory structure."""
import argparse
import os

from azureml.acft.common_components import get_logger_app

logger = get_logger_app("azureml.acft.contrib.hf.scripts.src.llm_evaluation.list_checkpoint_dirs")


def list_directory_contents(path, prefix="", max_depth=5, current_depth=0):
    """Recursively list directory contents with tree structure."""
    if current_depth >= max_depth:
        return

    try:
        items = sorted(os.listdir(path))
        for i, item in enumerate(items):
            item_path = os.path.join(path, item)
            is_last = i == len(items) - 1
            connector = "`-- " if is_last else "|-- "

            if os.path.isdir(item_path):
                logger.info(f"{prefix}{connector}{item}/")
                extension = "    " if is_last else "|   "
                list_directory_contents(item_path, prefix + extension, max_depth, current_depth + 1)
            else:
                size = os.path.getsize(item_path)
                size_str = format_size(size)
                logger.info(f"{prefix}{connector}{item} ({size_str})")
    except PermissionError:
        logger.info(f"{prefix}[Permission Denied]")
    except Exception as e:
        logger.info(f"{prefix}[Error: {str(e)}]")


def format_size(size):
    """Format file size in human readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def main():
    """Execute the script to list checkpoint directory structure."""
    parser = argparse.ArgumentParser(description="List checkpoint directory structure")
    parser.add_argument(
        "--checkpoint_base_path",
        type=str,
        required=True,
        help="Base path containing checkpoints"
    )
    parser.add_argument(
        "--max_depth",
        type=int,
        default=5,
        help="Maximum depth for directory traversal (default: 5)"
    )

    args = parser.parse_args()

    print("=" * 80)
    logger.info("Checkpoint Directory Structure Explorer")
    print("=" * 80)
    logger.info(f"\nBase Path: {args.checkpoint_base_path}")
    logger.info(f"Max Depth: {args.max_depth}")
    print("\n" + "=" * 80)

    data_path = args.checkpoint_base_path

    if not os.path.exists(data_path):
        logger.error(f"\nPath does not exist: {data_path}")
        return

    logger.info(f"\nPath exists: {data_path}")
    logger.info(f"Is Directory: {os.path.isdir(data_path)}")

    if os.path.isdir(data_path):
        logger.info("\n[Directory Contents]:\n")
        list_directory_contents(data_path, max_depth=args.max_depth)

        # Count files and directories
        total_files = 0
        total_dirs = 0
        total_size = 0

        for root, dirs, files in os.walk(data_path):
            total_dirs += len(dirs)
            total_files += len(files)
            for file in files:
                try:
                    total_size += os.path.getsize(os.path.join(root, file))
                except Exception:
                    # Ignore files that can't be accessed
                    pass

        print("\n" + "=" * 80)
        logger.info(r"Summary:")
        logger.info(f"  Total Directories: {total_dirs}")
        logger.info(f"  Total Files: {total_files}")
        logger.info(f"  Total Size: {format_size(total_size)}")
        print("=" * 80)
    else:
        size = os.path.getsize(data_path)
        logger.info(f"\n[File Size]: {format_size(size)}")
        print("=" * 80)


if __name__ == "__main__":
    main()
