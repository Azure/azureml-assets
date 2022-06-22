import argparse
import os
import shutil
import sys
import tempfile
import yaml
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List, Tuple

import azureml.assets as assets
import azureml.assets.environment as environment
import azureml.assets.util as util
from azureml.assets.util import logger

TASK_FILENAME = "_acr_build_task.yaml"
STEP_TIMEOUT_SECONDS = 60 * 90
SUCCESS_COUNT = "success_count"
FAILED_COUNT = "failed_count"
COUNTERS = [SUCCESS_COUNT, FAILED_COUNT]


def create_acr_task(image_name: str,
                    build_context_dir: Path,
                    dockerfile: str,
                    task_filename: str,
                    test_command: str = None,
                    push: bool = False):
    # Start task with just the build step
    task = {
        'version': 'v1.1.0',
        'stepTimeout': STEP_TIMEOUT_SECONDS,
        'steps': [{
            'id': "build",
            'build': f"-t $Registry/{image_name} -f {dockerfile} ."
        }]}

    # Add test command if provided
    if test_command:
        task['steps'].append({
            'id': 'test',
            'cmd': f"$Registry/{image_name} {test_command}"
        })

    # Add push step if requested
    if push:
        task['steps'].append({
            'id': 'push',
            'push': [f"$Registry/{image_name}"]
        })

    # Write YAML file to disk
    task_file = build_context_dir / task_filename
    with open(task_file, "w") as f:
        yaml.dump(task, f, default_flow_style=False, sort_keys=False)

    return task_file


def build_image(asset_config: assets.AssetConfig,
                image_name: str,
                build_context_dir: Path,
                dockerfile: str,
                build_log: Path,
                build_os: str = None,
                resource_group: str = None,
                registry: str = None,
                test_command: str = None,
                push: bool = False) -> Tuple[assets.AssetConfig, int, str]:
    logger.print(f"Building image for {asset_config.name}")
    start = timer()
    with tempfile.TemporaryDirectory() as temp_dir:
        if registry is not None:
            # Build on ACR
            cmd = ["az", "acr"]
            common_args = ["-g", resource_group, "-r", registry, "--platform", build_os]
            if not test_command and push:
                # Simple build and push
                cmd.append("build")
                cmd.extend(common_args)
                cmd.extend(["--file", dockerfile, "--image", image_name, "."])
            else:
                # Use ACR task from build context in temp dir
                temp_dir_path = Path(temp_dir)
                shutil.copytree(build_context_dir, temp_dir_path, dirs_exist_ok=True)
                build_context_dir = temp_dir_path
                create_acr_task(image_name, build_context_dir, dockerfile, TASK_FILENAME, test_command, push)
                cmd.append("run")
                cmd.extend(common_args)
                cmd.extend(["-f", TASK_FILENAME, "."])
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
                 registry: str = None,
                 test_command: str = None,
                 push: bool = False) -> bool:
    counters = Counter()
    with ThreadPoolExecutor(max_parallel) as pool:
        # Find environments under image root directories
        futures = []
        for asset_config in util.find_assets(input_dirs, asset_config_filename, assets.AssetType.ENVIRONMENT, changed_files):
            env_config = asset_config.environment_config_as_object()

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

            # Skip environments without build context
            if not env_config.build_enabled:
                logger.print(f"Skipping build of image for {asset_config.name}: No build context specified")

                # Copy file to output directory without building
                if output_directory:
                    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory, add_subdir=True)
                continue

            # Tag with version from spec
            if tag_with_version:
                version = asset_config.spec_as_object().version
                image_name = env_config.get_image_name_with_tag(version)
            else:
                image_name = env_config.image_name

            # Start building image
            build_log = build_logs_dir / f"{asset_config.name}.log"
            futures.append(pool.submit(build_image, asset_config, image_name,
                                       env_config.context_dir_with_path, env_config.dockerfile, build_log,
                                       env_config.os.value, resource_group, registry, test_command, push))

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
                    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory, add_subdir=True)

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
    parser.add_argument("-T", "--test-command", help="If building on ACR, command used to test image, relative to build context root")
    parser.add_argument("-u", "--push", action="store_true", help="If building on ACR, push after building and (optionally) testing")
    args = parser.parse_args()

    # Ensure dependent args are present
    if args.registry and not args.resource_group:
        parser.error("If --registry is specified then --resource-group is also required")
    if args.test_command and not (args.registry and args.resource_group):
        parser.error("--test-command requires both --registry and --resource-group")
    if args.push and not (args.registry and args.resource_group):
        parser.error("--push requires both --registry and --resource-group")

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
                           registry=args.registry,
                           test_command=args.test_command,
                           push=args.push)
    if not success:
        sys.exit(1)
