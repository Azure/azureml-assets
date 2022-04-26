import argparse
import os
import sys
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import timedelta
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List

from config import AssetConfig, AssetType, EnvironmentConfig, Os
from ci_logger import logger
from util import copy_asset_to_output_dir

SUCCESS_COUNT = "success_count"
FAILED_COUNT = "failed_count"
COUNTERS = [SUCCESS_COUNT, FAILED_COUNT]


def build_image(asset_config: AssetConfig, image_name: str, build_context_dir: str, dockerfile: str,
                build_log: str, build_os: str = None, resource_group: str = None, registry: str = None):
    print(f"Building {image_name}")
    start = timer()
    if registry is not None:
        # Build on ACR
        cmd = ["az", "acr", "build", "-g", resource_group, "-r", registry, "--file", dockerfile, "--platform", build_os, "--image", image_name, "."]
    else:
        # Build locally
        cmd = ["docker", "build", "--file", dockerfile, "--progress", "plain", "--tag", image_name, "."]
    p = run(cmd,
            cwd=build_context_dir,
            stdout=PIPE,
            stderr=STDOUT)
    end = timer()
    print(f"{image_name} built in {timedelta(seconds=end-start)}")
    os.makedirs(os.path.dirname(build_log), exist_ok=True)
    with open(build_log, "w") as f:
        f.write(p.stdout.decode())
    return (asset_config, image_name, p.returncode, p.stdout.decode())


# Doesn't support ACR yet
def get_image_digest(image_name: str):
    p = run(["docker", "image", "inspect", image_name, "--format=\"{{index .Id}}\""],
            stdout=PIPE,
            stderr=STDOUT)
    if p.returncode == 0:
        return p.stdout.decode()
    else:
        logger.log_warning(f"Failed to get image digest for {image_name}: {p.stdout.decode()}")
        return None


def build_images(input_dirs: List[str],
                 asset_config_filename: str,
                 output_directory: str,
                 build_logs_dir: str,
                 max_parallel: int,
                 changed_files: List[str],
                 os_to_build: str = None,
                 resource_group: str = None,
                 registry: str = None):
    changed_files_abs = [os.path.abspath(f) for f in changed_files]
    counters = Counter()
    with ThreadPoolExecutor(max_parallel) as pool:
        # Find environments under image root directories
        futures = []
        for input_dir in input_dirs:
            for root, _, files in os.walk(input_dir):
                for asset_config_file in [f for f in files if f == asset_config_filename]:
                    # Load config
                    asset_config = AssetConfig(os.path.join(root, asset_config_file))

                    # Skip if not environment
                    if asset_config.type is not AssetType.ENVIRONMENT:
                        continue
                    env_config = EnvironmentConfig(asset_config.extra_config_with_path)

                    # Filter by OS
                    if os_to_build and env_config.os.value != os_to_build:
                        print(f"Skipping build of {env_config.image_name}: Operating system {env_config.os.value} != {os_to_build}")
                        continue

                    # If provided, skip directories with no changed files
                    root_abs = os.path.abspath(root)
                    if changed_files and not any([f for f in changed_files_abs if f.startswith(f"{root_abs}/")]):
                        print(f"Skipping build of {env_config.image_name}: No files in its directory were changed")
                        continue

                    # Start building image
                    build_log = os.path.join(build_logs_dir, f"{env_config.image_name}.log")
                    futures.append(pool.submit(build_image, asset_config, env_config.image_name,
                                               env_config.context_dir_with_path, env_config.dockerfile, build_log,
                                               env_config.os.value, resource_group, registry))

        # Wait for builds to complete
        for future in as_completed(futures):
            (asset_config, image_name, return_code, output) = future.result()
            logger.start_group(f"{image_name} build log")
            print(output)
            logger.end_group()
            if return_code != 0:
                logger.log_error(f"Build of {image_name} failed with exit status {return_code}", "Build failure")
                counters[FAILED_COUNT] += 1
            else:
                logger.log_debug(f"Successfully built {image_name}")
                counters[SUCCESS_COUNT] += 1
                if output_directory:
                    copy_asset_to_output_dir(asset_config, output_directory)

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])

    if counters[FAILED_COUNT] > 0:
        logger.log_error(f"{counters[FAILED_COUNT]} environments failed to build")
        sys.exit(1)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing environments to build")
    parser.add_argument("-a", "--asset-config-filename", default="asset.yaml", help="Asset config file name to search for")
    parser.add_argument("-o", "--output-directory", help="Directory to which successfully built environments will be written")
    parser.add_argument("-l", "--build-logs-dir", required=True, help="Directory to receive build logs")
    parser.add_argument("-p", "--max-parallel", type=int, default=25, help="Maximum number of images to build at the same time")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter images")
    parser.add_argument("-O", "--os-to-build", choices=[i.value for i in list(Os)], help="Only build environments based on this OS")
    parser.add_argument("-r", "--registry", help="Container registry on which to build images")
    parser.add_argument("-g", "--resource-group", help="Resource group containing the container registry")
    args = parser.parse_args()

    # Ensure dependent args are present
    if args.registry and (not args.os_to_build or not args.resource_group):
        parser.error("If --registry is specified then --resource-group and --os-to-build are also required")

    # Convert comma-separated values to lists
    input_dirs = args.input_dirs.split(",")
    changed_files = args.changed_files.split(",") if args.changed_files else []

    # Build images
    build_images(input_dirs=input_dirs,
                 asset_config_filename=args.asset_config_filename,
                 output_directory=args.output_directory,
                 build_logs_dir=args.build_logs_dir,
                 max_parallel=args.max_parallel,
                 changed_files=changed_files,
                 os_to_build=args.os_to_build,
                 resource_group=args.resource_group,
                 registry=args.registry)
