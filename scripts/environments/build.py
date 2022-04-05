import argparse
import os
import sys
from ci_logger import logger
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import timedelta
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List

from env_config import EnvConfig, OS_OPTIONS
from pin_versions import transform

def build_image(image_name: str, build_context_dir: str, dockerfile: str, build_log: str,
                os_to_build: str=None, resource_group: str=None, registry: str=None):
    print(f"Building {image_name}")
    start = timer()
    if registry is not None:
        # Build on ACR
        cmd = ["az", "acr", "build", "-g", resource_group, "-r", registry, "--file", dockerfile, "--platform", os_to_build, "--image", image_name, "."]
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
    return (image_name, p.returncode, p.stdout.decode())

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

def create_github_env_var(key: str, value: str):
    # Make list of image names available to following steps
    github_env = os.getenv('GITHUB_ENV')
    if github_env:
        with open(github_env, "w+") as f:
            f.write(f"{key}={value}")
    else:
        logger.log_warning("Failed to write image names: GITHUB_ENV environment variable not found")

def build_images(image_dirs: List[str], env_config_filename: str, build_logs_dir: str,
                 max_parallel: int, changed_files: List[str], image_names_key: str,
                 os_to_build: str=None, resource_group: str=None, registry: str=None):
    with ThreadPoolExecutor(max_parallel) as pool:
        # Find Dockerfiles under image root directory
        futures = []
        for image_dir in image_dirs:
            for root, _, files in os.walk(image_dir):
                for env_config_file in [f for f in files if f == env_config_filename]:
                    # Load config
                    env_config = EnvConfig(os.path.join(root, env_config_file))

                    # Filter by OS
                    if os_to_build and env_config.os != os_to_build:
                        print(f"Skipping build of {env_config.image_name}: Operating system {env_config.os} != {os_to_build}")
                        continue

                    # If provided, skip directories with no changed files
                    if changed_files and not any([f for f in changed_files if f.startswith(f"{root}/")]):
                        print(f"Skipping build of {env_config.image_name}: No files in its directory were changed")
                        continue

                    # Pin images/packages in files
                    for file_to_pin in [os.path.join(root, env_config.context_dir, f) for f in env_config.pin_version_files]:
                        if os.path.exists(file_to_pin):
                            transform(file_to_pin)
                        else:
                            logger.log_warning(f"Failed to pin versions in {file_to_pin}: File not found")

                    # Start building image
                    build_log = os.path.join(build_logs_dir, f"{env_config.image_name}.log")
                    futures.append(pool.submit(build_image, env_config.image_name, os.path.join(root, env_config.context_dir),
                                               env_config.dockerfile, build_log, os_to_build, resource_group, registry))

        # Wait for builds to complete
        image_names = []
        for future in as_completed(futures):
            (image_name, return_code, output) = future.result()
            logger.start_group(f"{image_name} build log")
            print(output)
            logger.end_group()
            if return_code != 0:
                logger.log_error(f"Build of {image_name} failed with exit status {return_code}", "Build failure")
                sys.exit(1)
            image_names.append(image_name)
        
        # Make list of image names available to following steps
        create_github_env_var(image_names_key, ",".join(image_names))

if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dirs", required=True, help="Comma-separated list of directories containing image to build")
    parser.add_argument("-e", "--env-config-filename", default="env_config.py", help="Environment config file name to search for")
    parser.add_argument("-l", "--build-logs-dir", required=True, help="Directory to receive build logs")
    parser.add_argument("-p", "--max-parallel", type=int, default=25, help="Maximum number of images to build at the same time")
    parser.add_argument("-c", "--changed-files", help="Comma-separated list of changed files, used to filter images")
    parser.add_argument("-k", "--image-names-key", help="GitHub actions environment variable that will receive a comma-separated list of images built")
    parser.add_argument("-o", "--os-to-build", choices=OS_OPTIONS, help="Only build environments based on this OS")
    parser.add_argument("-g", "--resource-group", help="Resource group containing the container registry")
    parser.add_argument("-r", "--registry", help="Container registry on which to build images")
    args = parser.parse_args()

    # Ensure dependent args are present
    if args.registry and (not args.os_to_build or not args.resource_group):
        parser.error("If --registry is specified then --resource-group and --os-to-build are also required")

    # Convert comma-separated values to lists
    image_dirs = args.image_dirs.split(",")
    changed_files = args.changed_files.split(",") if args.changed_files else []
    
    # Build images
    build_images(image_dirs=image_dirs, env_config_filename=args.env_config_filename, build_logs_dir=args.build_logs_dir,
                 max_parallel=args.max_parallel, changed_files=changed_files, image_names_key=args.image_names_key,
                 os_to_build=args.os_to_build, resource_group=args.resource_group, registry=args.registry)

    