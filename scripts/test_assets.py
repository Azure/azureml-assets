import argparse
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from subprocess import run
from timeit import default_timer as timer
from typing import List

from ci_logger import logger
from config import AssetConfig
from util import find_assets

SUCCESS_COUNT = "success_count"
FAILED_COUNT = "failed_count"
COUNTERS = [SUCCESS_COUNT, FAILED_COUNT]
PYTEST_SPEC = "pytest==7.1.1"
BASE_ENVIRONMENT = "base_env"
ISOLATED_ENVIRONMENT = "isolated_env"


def create_isolated_environment(asset_config: AssetConfig, env_name: str) -> bool:
    print("Creating isolated conda environment")
    p = run(["conda", "create", "-n", env_name, "--clone", BASE_ENVIRONMENT, "-y", "-q"])
    if p.returncode != 0:
        return False

    print("Using pip to install packages")
    p = run(["conda", "run", "-n", env_name, "pip", "install", "-r", asset_config.pytest_pip_requirements,
             "--progress-bar", "off"], cwd=asset_config.file_path)
    return p.returncode == 0


def test_asset(asset_config: AssetConfig, env_name: str, reports_dir: str = None) -> bool:
    print("Running pytest")
    start = timer()

    # Build command line
    cmd = ["conda", "run", "-n", env_name, "pytest"]
    if reports_dir:
        report_file = reports_dir / asset_config.type.value / f"{asset_config.name}.xml"
        cmd.append(f"--junitxml={report_file}")
        cmd.extend(["-o", f"junit_suite_name={asset_config.type.value}/{asset_config.name}"])
    cmd.append(asset_config.pytest_tests_dir)

    # Run tests
    p = run(cmd, cwd=asset_config.file_path)
    end = timer()
    print(f"Test(s) completed in {timedelta(seconds=end-start)}")
    return p.returncode == 0


def test_assets(input_dirs: List[Path],
                asset_config_filename: str,
                changed_files: List[Path],
                reports_dir: Path = None):
    base_created = False
    counters = Counter()
    for asset_config in find_assets(input_dirs, asset_config_filename, changed_files=changed_files):
        # Skip assets without testing enabled
        if not asset_config.pytest_enabled:
            logger.log_debug(f"Testing is not enabled for {asset_config}")
            continue

        if not base_created:
            # Create base environment, which must succeed
            logger.start_group("Create base environment")
            run(["conda", "create", "-n", BASE_ENVIRONMENT, "-y", "-q", PYTEST_SPEC], check=True)
            base_created = True
            logger.end_group()

        logger.start_group(f"Test {asset_config}")
        success = True

        # Create isolated environment if packages will be installed
        test_env = BASE_ENVIRONMENT
        pytest_pip_requirements = asset_config.pytest_pip_requirements
        if pytest_pip_requirements:
            test_env = ISOLATED_ENVIRONMENT
            created = create_isolated_environment(asset_config, test_env)
            success = success and created

        # Run pytest
        if success:
            tested = test_asset(asset_config, test_env, reports_dir)
            success = success and tested

        counters[SUCCESS_COUNT if success else FAILED_COUNT] += 1
        logger.end_group()

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])

    if counters[FAILED_COUNT] > 0:
        logger.log_error(f"{counters[FAILED_COUNT]} asset(s) failed to test")
        sys.exit(1)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing environments to test")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter assets")
    parser.add_argument("-r", "--reports-dir", help="Directory for pytest reports")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Convert reports dir to Path
    reports_dir = Path(args.reports_dir) if args.reports_dir else None

    # Test assets
    test_assets(input_dirs=input_dirs,
                asset_config_filename=args.asset_config_filename,
                changed_files=changed_files,
                reports_dir=reports_dir)
