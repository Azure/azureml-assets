# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import sys
from collections import Counter
from datetime import timedelta
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List, Tuple

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

SUCCESS_COUNT = "success_count"
FAILED_COUNT = "failed_count"
COUNTERS = [SUCCESS_COUNT, FAILED_COUNT]
TEST_PHRASE = "hello world!"


def test_image(asset_config: assets.AssetConfig, image_name: str) -> Tuple[int, str]:
    logger.print(f"Testing image for {asset_config.name}")
    start = timer()
    p = run(["docker", "run", "--entrypoint", "python", image_name, "-c", f"print(\"{TEST_PHRASE}\")"],
            stdout=PIPE,
            stderr=STDOUT)
    end = timer()
    logger.print(f"Image for {asset_config.name} tested in {timedelta(seconds=end-start)}")
    return (p.returncode, p.stdout.decode())


def test_images(input_dirs: List[Path],
                asset_config_filename: str,
                output_directory: Path,
                os_to_test: str = None) -> bool:
    counters = Counter()
    for asset_config in util.find_assets(input_dirs, asset_config_filename, assets.AssetType.ENVIRONMENT):
        env_config = asset_config.environment_config_as_object()

        # Filter by OS
        if os_to_test and env_config.os.value != os_to_test:
            logger.print(f"Not testing image for {asset_config.name}: Operating system {env_config.os.value} != {os_to_test}")
            continue

        # Skip images without build context
        if not env_config.build_enabled:
            logger.print(f"Not testing image for {asset_config.name}: No build context specified")
            continue

        # Test image
        return_code, output = test_image(asset_config, env_config.image_name)
        if return_code != 0 or not output.startswith(TEST_PHRASE):
            logger.log_error(f"Testing of image for {asset_config.name} failed: {output}", title="Testing failure")
            counters[FAILED_COUNT] += 1
        else:
            logger.log_debug(f"Successfully tested image for {asset_config.name}")
            counters[SUCCESS_COUNT] += 1
            if output_directory:
                util.copy_asset_to_output_dir(asset_config=asset_config, output_dir=output_directory, add_subdir=True)

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])

    if counters[FAILED_COUNT] > 0:
        logger.log_error(f"{counters[FAILED_COUNT]} environment image(s) failed to test")
        return False
    return True


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing environments to test")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    parser.add_argument("-o", "--output-directory", type=Path, help="Directory to which successfully tested environments will be written")
    parser.add_argument("-O", "--os-to-test", choices=[i.value for i in list(assets.Os)], help="Only test environments based on this OS")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]

    # Test images
    success = test_images(input_dirs=input_dirs,
                          asset_config_filename=args.asset_config_filename,
                          output_directory=args.output_directory,
                          os_to_test=args.os_to_test)
    if not success:
        sys.exit(1)
