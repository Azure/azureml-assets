# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Setup script for the package."""

from setuptools import setup, find_packages

# Read requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()


setup(
    name="fm-optimized-inference",
    version="0.1",
    author="Microsoft",
    author_email="",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
        ],
    },
    python_requires=">=3.8",
)
