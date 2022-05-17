import argparse
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List, Tuple

from ci_logger import logger
from config import AssetConfig, AssetType, EnvironmentConfig, Os
from util import copy_asset_to_output_dir, find_assets

SUCCESS_COUNT = "success_count"
FAILED_COUNT = "failed_count"
COUNTERS = [SUCCESS_COUNT, FAILED_COUNT]
TEST_PHRASE = "hello world!"
BASE_ENVIRONMENT = "base_env"
ISOLATED_ENVIRONMENT = "isolated_env"


def test_image(asset_config: AssetConfig, image_name: str) -> Tuple[int, str]:
    print(f"Testing image for {asset_config.name}")
    start = timer()
    p = run(["docker", "run", "--entrypoint", "python", image_name, "-c", f"print(\"{TEST_PHRASE}\")"],
            stdout=PIPE,
            stderr=STDOUT)
    end = timer()
    print(f"Image for {asset_config.name} tested in {timedelta(seconds=end-start)}")
    return (p.returncode, p.stdout.decode())


def test_assets(input_dirs: List[Path],
                asset_config_filename: str,
                changed_files: List[Path]):
    base_created = False
    for asset_config in find_assets(input_dirs, asset_config_filename, changed_files=changed_files):
        # Skip assets without testing enabled
        if not asset_config.test_enabled:
            logger.log_debug(f"Testing is not enabled for {asset_config}")
            continue

        print(f"Testing {asset_config}")

        if not base_created:
            # Create base environment, which must succeed
            run(["conda", "create", "-n", BASE_ENVIRONMENT, "-y", "-q", "pytest"], check=True)
            base_created = True

        test_env = BASE_ENVIRONMENT
        pip_requirements = asset_config.test_pip_requirements
        if pip_requirements:
            test_env = ISOLATED_ENVIRONMENT
            print("Creating isolated conda environment")
            p = run(["conda", "create", "-n", test_env, "--clone", BASE_ENVIRONMENT, "-y", "-q"])
            # TODO: Check p
            print("Using pip to install packages")
            p = run(["conda", "run", "-n", test_env, "pip", "install", "-r", pip_requirements, "-q"], cwd=asset_config.file_path)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing environments to test")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter assets")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Test assets
    test_assets(input_dirs=input_dirs,
                asset_config_filename=args.asset_config_filename,
                changed_files=changed_files)
