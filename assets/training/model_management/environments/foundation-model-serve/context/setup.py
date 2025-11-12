# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Setup script for the package."""
import os
import re
from typing import Any, Match, cast

from setuptools import setup, find_packages
from pathlib import Path

base_dir = Path(__file__).resolve().parent
requirements_file = base_dir / "requirements.txt"

with open(
    requirements_file,
    encoding="utf-8",
) as f:
    requirements = f.read().splitlines()

PACKAGE_FOLDER_PATH = "foundation/model/serve"

# Version extraction inspired from 'requests'
with open(
    os.path.join(PACKAGE_FOLDER_PATH, "_version.py"),
    "r",
    encoding="utf-8",
) as fd:
    version = cast(
        Match[Any],
        re.search(
            r'^VERSION\s*=\s*[\'"]([^\'"]*)[\'"]',
            fd.read(),
            re.MULTILINE,
        ),
    ).group(1)

if not version:
    raise RuntimeError("Cannot find version information")

setup(
    name="foundation-model-serve",
    version=version,
    author="Microsoft",
    author_email="",
    description="",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(exclude=["*tests*"]),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    install_requires=requirements,
    package_data={
        "foundation.model.serve": ["api_server_setup/*.json"],
    },
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.9",
)
