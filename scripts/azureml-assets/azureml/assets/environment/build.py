import argparse
import os
import sys
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
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


def build_image(asset_config: assets.AssetConfig,
                image_name: str,
                build_context_dir: Path,
                dockerfile: str,
                build_log: Path,
                build_os: str = None,
                resource_group: str = None,
                registry: str = None) -> Tuple[assets.AssetConfig, int, str]:
    logger.print(f"Building image for {asset_config.name}")
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
    logger.print(f"Image for {asset_config.name} built in {timedelta(seconds=end-start)}")
    os.makedirs(build_log.parent, exist_ok=True)
    with open(build_log, "w") as f:
        f.write(p.stdout.decode())
    return (asset_config, p.returncode, p.stdout.decode())


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


def build_images(input_dirs: List[Path],
                 asset_config_filename: str,
                 output_directory: Path,
                 build_logs_dir: Path,
                 pin_versions: bool,
                 max_parallel: int,
                 changed_files: List[Path],
                 tag_with_version: bool,
                 os_to_build: str = None,
                 resource_group: str = None,
                 registry: str = None) -> bool:
    counters = Counter()
    with ThreadPoolExecutor(max_parallel) as pool:
        # Find environments under image root directories
        futures = []
        for asset_config in util.find_assets(input_dirs, asset_config_filename, assets.AssetType.ENVIRONMENT, changed_files):
            env_config = assets.EnvironmentConfig(asset_config.extra_config_with_path)

            # Filter by OS
            if os_to_build and env_config.os.value != os_to_build:
                logger.print(f"Skipping build of image for {asset_config.name}: Operating system {env_config.os.value} != {os_to_build}")
                continue

            # Pin versions
            if pin_versions:
                try:
                    assets.pin_env_files(env_config)
                except Exception as e:
                    logger.log_error(f"Failed to pin versions for {asset_config.name}: {e}")
                    counters[FAILED_COUNT] += 1
                    continue

            # Tag with version from spec
            if tag_with_version:
                version = assets.Spec(asset_config.spec_with_path).version
                image_name = env_config.get_image_name_with_tag(version)
            else:
                image_name = env_config.image_name

            # Start building image
            build_log = build_logs_dir / f"{asset_config.name}.log"
            futures.append(pool.submit(build_image, asset_config, image_name,
                                       env_config.context_dir_with_path, env_config.dockerfile, build_log,
                                       env_config.os.value, resource_group, registry))

        # Wait for builds to complete
        for future in as_completed(futures):
            (asset_config, return_code, output) = future.result()
            logger.start_group(f"{asset_config.name} build log")
            logger.print(output)
            logger.end_group()
            if return_code != 0:
                logger.log_error(f"Build of image for {asset_config.name} failed with exit status {return_code}", "Build failure")
                counters[FAILED_COUNT] += 1
            else:
                logger.log_debug(f"Successfully built image for {asset_config.name}")
                counters[SUCCESS_COUNT] += 1
                if output_directory:
                    util.copy_asset_to_output_dir(asset_config, output_directory)

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])

    if counters[FAILED_COUNT] > 0:
        logger.log_error(f"{counters[FAILED_COUNT]} environment image(s) failed to build")
        return False
    return True


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True, help="Comma-separated list of directories containing environments to build")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME, help="Asset config file name to search for")
    parser.add_argument("-o", "--output-directory", type=Path, help="Directory to which successfully built environments will be written")
    parser.add_argument("-l", "--build-logs-dir", required=True, type=Path, help="Directory to receive build logs")
    parser.add_argument("-p", "--max-parallel", type=int, default=25, help="Maximum number of images to build at the same time")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter images")
    parser.add_argument("-O", "--os-to-build", choices=[i.value for i in list(assets.Os)], help="Only build environments based on this OS")
    parser.add_argument("-r", "--registry", help="Container registry on which to build images")
    parser.add_argument("-g", "--resource-group", help="Resource group containing the container registry")
    parser.add_argument("-P", "--pin-versions", action="store_true", help="Pin images/packages to latest versions")
    parser.add_argument("-t", "--tag-with-version", action="store_true", help="Tag image names using the version in the asset's spec file")
    args = parser.parse_args()

    # Ensure dependent args are present
    if args.registry and not args.resource_group:
        parser.error("If --registry is specified then --resource-group is also required")

    # Convert comma-separated values to lists
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []

    # Build images
    success = build_images(input_dirs=input_dirs,
                           asset_config_filename=args.asset_config_filename,
                           output_directory=args.output_directory,
                           build_logs_dir=args.build_logs_dir,
                           pin_versions=args.pin_versions,
                           max_parallel=args.max_parallel,
                           changed_files=changed_files,
                           tag_with_version=args.tag_with_version,
                           os_to_build=args.os_to_build,
                           resource_group=args.resource_group,
                           registry=args.registry)
    if not success:
        sys.exit(1)
