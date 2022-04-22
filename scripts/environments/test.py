import argparse
import os
from datetime import timedelta
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List

from ci_logger import logger
from config import AssetConfig, AssetType, EnvironmentConfig, Os
from util import copy_asset_to_output_dir

TEST_PHRASE = "hello world!"


def test_image(image_name: str):
    print(f"Testing {image_name}")
    start = timer()
    p = run(["docker", "run", image_name, "python", "-c", f"print(\"{TEST_PHRASE}\")"],
            stdout=PIPE,
            stderr=STDOUT)
    end = timer()
    print(f"{image_name} tested in {timedelta(seconds=end-start)}")
    return (p.returncode, p.stdout.decode())


def test_images(image_dirs: List[str],
                asset_config_filename: str,
                output_directory: str,
                os_to_test: str = None):
    for image_dir in image_dirs:
        for root, _, files in os.walk(image_dir):
            for asset_config_file in [f for f in files if f == asset_config_filename]:
                # Load config
                asset_config = AssetConfig(os.path.join(root, asset_config_file))

                # Skip if not environment
                if asset_config.type is not AssetType.ENVIRONMENT:
                    continue
                env_config = EnvironmentConfig(asset_config.extra_config_with_path)

                # Filter by OS
                if os_to_test and env_config.os.value != os_to_test:
                    print(f"Not testing {env_config.image_name}: Operating system {env_config.os.value} != {os_to_test}")
                    continue

                # Test image
                (return_code, output) = test_image(env_config.image_name)
                if return_code != 0 or not output.startswith(TEST_PHRASE):
                    logger.log_error(f"Test failure on {env_config.image_name}: {output}", title="Testing failure")
                else:
                    logger.log_debug(f"Test successful on {env_config.image_name}")
                    if output_directory:
                        copy_asset_to_output_dir(asset_config, output_directory)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing environments to test")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-o", "--output-directory", help="Directory to which successfully tested environments will be written")
    parser.add_argument("-O", "--os-to-test", choices=[i.value for i in list(Os)], help="Only test environments based on this OS")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    image_dirs = args.image_dirs.split(",")

    # Test images
    test_images(image_dirs=image_dirs,
                asset_config_filename=args.asset_config_filename,
                output_directory=args.output_directory,
                os_to_test=args.os_to_test)
