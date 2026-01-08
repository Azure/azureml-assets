# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Build environment images."""

import argparse
import os
import shutil
import sys
import tempfile
from collections import Counter
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import timedelta, datetime
from pathlib import Path
from ruamel.yaml import YAML
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List, Tuple

import azureml.assets as assets
import azureml.assets.util as util
from azureml.assets.util import logger

TASK_FILENAME = "_acr_build_task.yaml"
BUILD_STEP_TIMEOUT_SECONDS = 60 * 420  # 7 hours - leave room for other steps (max total is 8 hours)
SCAN_STEP_TIMEOUT_SECONDS = 60 * 10  # Reduced to 10 minutes
DEFAULT_STEP_TIMEOUT_SECONDS = 60 * 10  # Reduced to 10 minutes
TRIVY_TIMEOUT = "15m0s"
TRIVY_VERSION = "v0.64.1-1"
ORAS_VERSION = "v1.2.3-2"
SUCCESS_COUNT = "success_count"
FAILED_COUNT = "failed_count"
COUNTERS = [SUCCESS_COUNT, FAILED_COUNT]
BUILT_IMAGES = "built_images"


def create_acr_task(image_name: str,
                    dockerfile: str,
                    os: assets.Os,
                    task_filename: str,
                    test_command: str = None,
                    push: bool = False,
                    trivy_url: str = None) -> int:
    """Create ACR build task file.

    Args:
        image_name (str): Image name.
        dockerfile (str): Dockerfile name.
        os (assets.Os): Operating system of the image.
        task_filename (str): File to which the task will be written.
        test_command (str, optional): Command used to test the image. Defaults to None.
        push (bool, optional): Push the image to the ACR. Defaults to False.
        trivy_url (str, optional): URL to download Trivy for vulnerability scanning. Defaults to None.

    Returns:
        int: Cumulative total of step timeouts, to be used when creating the ACR task.
    """
    # Start task with just the build step
    task = {
        'version': 'v1.1.0',
        'stepTimeout': DEFAULT_STEP_TIMEOUT_SECONDS,
        'steps': [{
            'id': "build",
            'timeout': BUILD_STEP_TIMEOUT_SECONDS,
            'build': f"-t $Registry/{image_name} -f {dockerfile} ."
        }]}

    # Add command to output packages
    if os is assets.Os.LINUX:
        # Quoted string required to handle && and ||
        cmd = r'/bin/sh -c "[ -n \"\$CONDA_DEFAULT_ENV\" ] && conda env export || pip freeze"'
    else:
        # Requires Windows batch
        cmd = r'cmd /C "if defined CONDA_DEFAULT_ENV (conda env export) else (pip freeze)"'
    task['steps'].append({
        'id': "package_export",
        'cmd': f"$Registry/{image_name} {cmd}",
        'ignoreErrors': True
    })

    # Add test command if provided
    if test_command:
        task['steps'].append({
            'id': "test",
            'cmd': f"$Registry/{image_name} {test_command}"
        })

    # Add vulnerability scanning step if requested
    if trivy_url is not None:
        if os is assets.Os.LINUX:
            scan_cmd = (
                f"mcr.microsoft.com/oss/v2/aquasecurity/trivy:{TRIVY_VERSION} "
                f"-q --timeout {TRIVY_TIMEOUT} image --scanners vuln --ignore-unfixed "
                f"$Registry/{image_name}"
            )
            task['steps'].append({
                'id': "scan",
                'timeout': SCAN_STEP_TIMEOUT_SECONDS,
                'cmd': scan_cmd,
                'ignoreErrors': True,
                'privileged': True
            })
        else:
            logger.log_warning(f"Skipped vulnerability scan for Windows image {image_name} - not supported")

    # Add push step if requested
    if push:
        task['steps'].append({
            'id': "push",
            'push': [f"$Registry/{image_name}"]
        })
        # Add post-push vulnerability scan and SBOM attachment steps
        post_scan_cmd = (
            f"mcr.microsoft.com/oss/v2/aquasecurity/trivy:{TRIVY_VERSION} "
            f"image --no-progress --format spdx-json --output sbom.json "
            f"$Registry/{image_name} --timeout 15m0s"
        )
        task['steps'].append({
            'id': "scan",
            'timeout': 600,  # 10 minute timeout for scanning
            'cmd': post_scan_cmd,
            'ignoreErrors': True,  # Don't fail build if scan fails
            'privileged': True     # Required for some scan operations
        })
        # Compute current UTC timestamp in ISO 8601 format
        timestamp = datetime.utcnow().isoformat() + "Z"
        # Attach SBOM to the image
        oras_cmd = (
            f"mcr.microsoft.com/oss/v2/oras-project/oras:{ORAS_VERSION} attach $Registry/{image_name} "
            f"--artifact-type application/vnd.microsoft.artifact.sbom+json "
            f"--annotation 'org.opencontainers.image.created={timestamp}' "
            f"--annotation 'org.opencontainers.image.title=SBOM generated by Trivy' "
            f"--annotation 'org.opencontainers.image.description=Software Bill of Materials generated by Trivy' "
            f"sbom.json:application/json"
        )
        task['steps'].append({
            'id': "export-image",
            'timeout': 180,  # 3 minute timeout for SBOM attachment
            'cmd': oras_cmd,
            'ignoreErrors': True  # Don't fail build if SBOM attachment fails
        })

    # Write YAML file to disk
    with open(task_filename, "w", encoding='utf-8') as f:
        yaml = YAML()
        yaml.default_flow_style = False
        yaml.dump(task, f)

    # Compute task timeout by adding all step timeouts
    return sum([step.get('timeout', DEFAULT_STEP_TIMEOUT_SECONDS) for step in task['steps']])


def build_image(asset_config: assets.AssetConfig,
                env_config: assets.EnvironmentConfig,
                image_name: str,
                build_log: Path,
                resource_group: str = None,
                registry: str = None,
                test_command: str = None,
                push: bool = False,
                trivy_url: str = None) -> Tuple[assets.AssetConfig, str, int, str]:
    """Build a Docker image locally or via ACR task.

    Args:
        asset_config (assets.AssetConfig): Asset config.
        env_config (assets.EnvironmentConfig): Environment config.
        image_name (str): Image name.
        build_log (Path): File to which the build log will be written.
        resource_group (str, optional): Resource group of the ACR. Defaults to None.
        registry (str, optional): ACR name. Defaults to None.
        test_command (str, optional): Command used to test the image. Defaults to None.
        push (bool, optional): Push the image to the ACR. Defaults to False.
        trivy_url (str, optional): URL to download Trivy for vulnerability scanning. Defaults to None.

    Returns:
        Tuple[assets.AssetConfig, str, int, str]: Asset config, image name, return code, and contents of stdout.
    """
    logger.print(f"Building image for {asset_config.name}")
    start = timer()
    with tempfile.TemporaryDirectory() as temp_dir:
        if registry is not None:
            # Build via ACR task
            temp_dir_path = Path(temp_dir)
            shutil.copytree(env_config.context_dir_with_path, temp_dir_path, dirs_exist_ok=True)
            build_context_dir = temp_dir_path
            task_timeout = create_acr_task(image_name=image_name, dockerfile=env_config.dockerfile,
                                           os=env_config.os, task_filename=build_context_dir / TASK_FILENAME,
                                           test_command=test_command, push=push, trivy_url=trivy_url)
            cmd = ["az", "acr", "run", "-g", resource_group, "-r", registry, "--platform", env_config.os.value,
                   "--timeout", str(task_timeout), "-f", TASK_FILENAME, "."]
        else:
            # Build locally
            build_context_dir = env_config.context_dir_with_path
            cmd = ["docker", "build", "--file", env_config.dockerfile, "--progress", "plain", "--tag", image_name, "."]
        p = run(cmd,
                cwd=build_context_dir,
                stdout=PIPE,
                stderr=STDOUT)
    end = timer()
    logger.print(f"Image for {asset_config.name} built in {timedelta(seconds=end-start)}")
    os.makedirs(build_log.parent, exist_ok=True)
    with open(build_log, "w", encoding='utf-8') as f:
        f.write(p.stdout.decode())
    return (asset_config, image_name, p.returncode, p.stdout.decode())


# Doesn't support ACR yet
def get_image_digest(image_name: str) -> str:
    """Get image digest for a local image.

    Args:
        image_name (str): Image name.

    Returns:
        str: Image digest.
    """
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
                 os_to_build: assets.Os = None,
                 resource_group: str = None,
                 registry: str = None,
                 test_command: str = None,
                 push: bool = False,
                 use_version_dirs: bool = False,
                 trivy_url: str = None) -> bool:
    """Build Docker images in parallel, either locally or via ACR.

    Args:
        input_dirs (List[Path]): Input directories where environments are located.
        asset_config_filename (str): Asset config filename to search for.
        output_directory (Path): Directory to which assets successfully built will be copied.
        build_logs_dir (Path): Directory to receive build logs.
        pin_versions (bool): Pin image versions.
        max_parallel (int): Maximum number of images to build in parallel.
        changed_files (List[Path]): List of changed files, used to filter environments.
        tag_with_version (bool): Tag image with asset version.
        os_to_build (assets.Os, optional): Operating system to build on via ACR. Defaults to None.
        resource_group (str, optional): Resource group name for ACR builds. Defaults to None.
        registry (str, optional): ACR name. Defaults to None.
        test_command (str, optional): Command used to test images. Defaults to None.
        push (bool, optional): Push images to ACR. Defaults to False.
        use_version_dirs (bool, optional): Use version directories for output. Defaults to False.
        trivy_url (str, optional): URL to download Trivy for vulnerability scanning. Defaults to None.

    Returns:
        bool: True if all images were build successfully, otherwise False.
    """
    counters = Counter()
    with ThreadPoolExecutor(max_parallel) as pool:
        # Find environments under image root directories
        futures = []
        for asset_config in util.find_assets(input_dirs, asset_config_filename, assets.AssetType.ENVIRONMENT,
                                             changed_files):
            env_config = asset_config.extra_config_as_object()

            # Filter by OS
            if os_to_build and env_config.os != os_to_build:
                logger.print(f"Skipping build of image for {asset_config.name}: "
                             f"Operating system {env_config.os.value} != {os_to_build.value}")
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
                    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory,
                                                  add_subdir=True, use_version_dir=use_version_dirs)
                continue

            # Push image for vulnerability scanning
            push_this_image = push

            # Tag with version from spec
            if tag_with_version:
                version = asset_config.version
                image_name = env_config.get_image_name_with_tag(version)
            else:
                image_name = env_config.image_name

            # Start building image
            build_log = build_logs_dir / f"{asset_config.name}.log"
            futures.append(pool.submit(build_image, asset_config=asset_config, env_config=env_config,
                                       image_name=image_name, build_log=build_log, resource_group=resource_group,
                                       registry=registry, test_command=test_command, push=push_this_image,
                                       trivy_url=trivy_url))

        # Wait for builds to complete
        built_images = []
        for future in as_completed(futures):
            (asset_config, image_name, return_code, output) = future.result()
            logger.start_group(f"{asset_config.name} build log")
            logger.print(output)
            logger.end_group()
            if return_code != 0:
                logger.log_error(f"Build of image for {asset_config.name} failed with exit status {return_code}",
                                 "Build failure")
                counters[FAILED_COUNT] += 1
            else:
                logger.log_debug(f"Successfully built image for {asset_config.name}")
                counters[SUCCESS_COUNT] += 1
                built_images.append(image_name)
                if output_directory:
                    util.copy_asset_to_output_dir(asset_config=asset_config, output_directory=output_directory,
                                                  add_subdir=True, use_version_dir=use_version_dirs)

    # Set variables
    for counter_name in COUNTERS:
        logger.set_output(counter_name, counters[counter_name])
    logger.set_output(BUILT_IMAGES, ",".join(built_images))

    if counters[FAILED_COUNT] > 0:
        logger.log_error(f"{counters[FAILED_COUNT]} environment image(s) failed to build")
        return False
    return True


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input-dirs", required=True,
                        help="Comma-separated list of directories containing environments to build")
    parser.add_argument("-a", "--asset-config-filename", default=assets.DEFAULT_ASSET_FILENAME,
                        help="Asset config file name to search for")
    parser.add_argument("-o", "--output-directory", type=Path,
                        help="Directory to which successfully built environments will be written")
    parser.add_argument("-l", "--build-logs-dir", required=True, type=Path,
                        help="Directory to receive build logs")
    parser.add_argument("-p", "--max-parallel", type=int, default=25,
                        help="Maximum number of images to build at the same time")
    parser.add_argument("-c", "--changed-files",
                        help="Comma-separated list of changed files, used to filter images")
    parser.add_argument("-O", "--os-to-build", choices=[i.value for i in list(assets.Os)],
                        help="Only build environments based on this OS")
    parser.add_argument("-r", "--registry",
                        help="Container registry on which to build images")
    parser.add_argument("-g", "--resource-group",
                        help="Resource group containing the container registry")
    parser.add_argument("-P", "--pin-versions", action="store_true",
                        help="Pin images/packages to latest versions")
    parser.add_argument("-t", "--tag-with-version", action="store_true",
                        help="Tag image names using the version in the asset's spec file")
    parser.add_argument("-T", "--test-command",
                        help="If building on ACR, command used to test image, relative to build context root")
    parser.add_argument("-u", "--push", action="store_true",
                        help="If building on ACR, push after building and (optionally) testing")
    parser.add_argument("-v", "--use-version-dirs", action="store_true",
                        help="Use version directories when storing assets in output directory")
    parser.add_argument("-U", "--trivy-url", type=str,
                        help="URL to download Trivy to scan for vulnerabilities")
    args = parser.parse_args()

    # Ensure dependent args are present
    if args.registry and not args.resource_group:
        parser.error("If --registry is specified then --resource-group is also required")
    if args.test_command and not (args.registry and args.resource_group):
        parser.error("--test-command requires both --registry and --resource-group")
    if args.push and not (args.registry and args.resource_group):
        parser.error("--push requires both --registry and --resource-group")
    if args.use_version_dirs and not args.output_directory:
        parser.error("--use-version-dirs requires --output-directory")

    # Reformat arg values
    input_dirs = [Path(d) for d in args.input_dirs.split(",")]
    changed_files = [Path(f) for f in args.changed_files.split(",")] if args.changed_files else []
    os_to_build = assets.Os(args.os_to_build) if args.os_to_build else None

    # Build images
    success = build_images(input_dirs=input_dirs,
                           asset_config_filename=args.asset_config_filename,
                           output_directory=args.output_directory,
                           build_logs_dir=args.build_logs_dir,
                           pin_versions=args.pin_versions,
                           max_parallel=args.max_parallel,
                           changed_files=changed_files,
                           tag_with_version=args.tag_with_version,
                           os_to_build=os_to_build,
                           resource_group=args.resource_group,
                           registry=args.registry,
                           test_command=args.test_command,
                           push=args.push,
                           use_version_dirs=args.use_version_dirs,
                           trivy_url=args.trivy_url)
    if not success:
        sys.exit(1)
