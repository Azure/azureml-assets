import argparse
import os
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import timedelta
from pin_versions import transform
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List

def build_image(image_name: str, build_context_dir: str, dockerfile: str, build_log: str):
    print(f"Building {image_name}")
    start = timer()
    p = run(["docker", "build", "--file", dockerfile, "--progress", "plain", "--tag", image_name, "."],
            cwd=build_context_dir,
            stdout=PIPE,
            stderr=STDOUT)
    end = timer()
    print(f"{image_name} built in {timedelta(seconds=end-start)}")
    with open(build_log, "w") as f:
        f.write(p.stdout.decode())
    return (image_name, p.returncode, p.stdout.decode())

def get_image_digest(image_name: str):
    p = run(["docker", "image", "inspect", image_name, "--format=\"{{index .Id}}\""],
            stdout=PIPE,
            stderr=STDOUT)
    if p.returncode == 0:
        return p.stdout.decode()
    else:
        print(f"::warning Failed to get image digest for {image_name}: {p.stdout.decode()}")
        return None

def create_github_env_var(key: str, value: str):
    # Make list of image names available to following steps
        github_env = os.getenv('GITHUB_ENV')
        if github_env:
            with open(github_env, "w+") as f:
                f.write(f"{key}={value}")
        else:
            print("::warning Failed to write image names: GITHUB_ENV environment variable not found")

def build_images(image_dirs: List[str], dockerfile_name: str, build_logs_dir: str, max_parallel: int,
                 changed_files: List[str], image_names_key: str, files_to_pin: List[str]):
    with ThreadPoolExecutor(max_parallel) as pool:
        # Find Dockerfiles under image root directory
        futures = []
        for image_dir in image_dirs:
            for root, _, files in os.walk(image_dir):
                for dockerfile in [f for f in files if f == dockerfile_name]:
                    # If provided, skip directories with no changed files
                    image_name = os.path.basename(root).lower()
                    if changed_files and not any([f for f in changed_files if f.startswith(f"{root}/")]):
                        print(f"Skipping build of {image_name}: No files in its directory were changed")
                        continue

                    # Pin images/packages in files
                    for file_to_pin in [os.path.join(root, f) for f in files_to_pin]:
                        if os.path.exists(file_to_pin):
                            transform(file_to_pin)

                    # Start building image
                    build_log = os.path.join(build_logs_dir, f"{image_name}.log")
                    futures.append(pool.submit(build_image, image_name, root, dockerfile, build_log))

        # Wait for builds to complete
        image_names = []
        for future in as_completed(futures):
            (image_name, return_code, output) = future.result()
            print(f"::group::{image_name} build log")
            print(output)
            print("::endgroup::")
            if return_code != 0:
                print(f"::error title=Build failure::Build of {image_name} failed with exit status {return_code}")
                sys.exit(1)
            image_names.append(image_name)

            digest = get_image_digest(image_name)
        
        # Make list of image names available to following steps
        create_github_env_var(image_names_key, ",".join(image_names))

if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-dirs", required=True, help="Comma-separated list of directories containing image to build")
    parser.add_argument("-d", "--dockerfile-name", default="Dockerfile", help="Dockerfile names to search for")
    parser.add_argument("-l", "--build-logs-dir", required=True, help="Directory to receive build logs")
    parser.add_argument("-p", "--max-parallel", type=int, default=25, help="Maximum number of images to build at the same time")
    parser.add_argument("-c", "--changed_files", help="Comma-separated list of changed files, used to filter images")
    parser.add_argument("-k", "--image-names-key", help="GitHub actions environment variable that will receive a comma-separated list of images built")
    parser.add_argument("-P", "--files-to-pin", help="Comma-separated list of files that should be updated to pin images/packages to latest versions")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    image_dirs = args.image_dirs.split(",")
    changed_files = args.changed_files.split(",") if args.changed_files else []
    files_to_pin = args.files_to_pin.split(",") if args.files_to_pin else []

    # Build images
    build_images(image_dirs, args.dockerfile_name, args.build_logs_dir, args.max_parallel, changed_files, args.image_names_key, files_to_pin)

    