# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

import argparse
import sys
from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List

from azureml.assets.util import logger

TAG_DATETIME_FORMAT = "%Y%m%d%H%M%S"


def tag_image(image_name: str, target_image_name: str):
    p = run(["docker", "tag", image_name, target_image_name],
            stdout=PIPE,
            stderr=STDOUT)
    return (p.returncode, p.stdout.decode())


def push_image(image_name: str, all_tags: bool = False):
    # Create args
    args = ["docker", "push"]
    if all_tags:
        args.append("--all-tags")
    args.append(image_name)

    # Push
    logger.print(f"Pushing {image_name}")
    start = timer()
    p = run(args,
            stdout=PIPE,
            stderr=STDOUT)
    end = timer()
    logger.print(f"{image_name} pushed in {timedelta(seconds=end-start)}")
    return (image_name, p.returncode, p.stdout.decode())


def push_images(image_names: List[str], target_image_prefix: str, tags: List[str], max_parallel: int):
    with ThreadPoolExecutor(max_parallel) as pool:
        futures = []
        for image_name in image_names:
            # Tag image
            target_image_base_name = f"{target_image_prefix}{image_name}"
            logger.print(f"Tagging {target_image_base_name} as {tags}")
            for tag in tags:
                target_image_name = f"{target_image_base_name}:{tag}"
                (return_code, output) = tag_image(image_name, target_image_name)
                if return_code != 0:
                    logger.log_error(f"Failed to tag {image_name} as {target_image_name}: {output}", "Tag failure")
                    sys.exit(1)

            # Start push
            futures.append(pool.submit(push_image, target_image_base_name, True))

        for future in as_completed(futures):
            (image_name, return_code, output) = future.result()
            logger.start_group(f"{image_name} push log")
            logger.print(output)
            logger.end_group()
            if return_code != 0:
                logger.log_error(f"Push of {image_name} failed with exit status {return_code}", "Push failure")
                sys.exit(1)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-names", required=True, help="Comma-separated list of image names to push")
    parser.add_argument("-t", "--target-image-prefix", required=True, help="Prefix to use when tagging images")
    parser.add_argument("-p", "--max-parallel", type=int, default=5, help="Maximum number of images to push at the same time")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    image_names = args.image_names.split(",") if args.image_names else []

    # Generate tags
    timestamp_tag = datetime.now(timezone.utc).strftime(TAG_DATETIME_FORMAT)
    tags = [timestamp_tag, "latest"]

    # Push images
    push_images(image_names, args.target_image_prefix, tags, args.max_parallel)
