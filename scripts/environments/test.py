import argparse
import sys
from datetime import timedelta
from subprocess import run, PIPE, STDOUT
from timeit import default_timer as timer
from typing import List

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


def test_images(image_names: List[str]):
    for image_name in image_names:
        # Test image
        (return_code, output) = test_image(image_name)
        if return_code != 0 or not output.startswith(TEST_PHRASE):
            print(f"::error title=Testing failure::Failed to test {image_name}: {output}")
            sys.exit(1)


if __name__ == '__main__':
    # Handle command-line args
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image-names", required=True, help="Comma-separated list of image names to test")
    args = parser.parse_args()

    # Convert comma-separated values to lists
    image_names = args.image_names.split(",") if args.image_names else []

    # Test images
    test_images(image_names)
