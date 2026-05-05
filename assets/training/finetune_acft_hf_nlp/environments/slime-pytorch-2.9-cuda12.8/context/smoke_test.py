# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import pathlib
import zipfile

from packaging.version import Version
import PIL
import ray
import sglang
import slime
import torch


LOG4J_ARTIFACTS = ("log4j-api", "log4j-core", "log4j-slf4j-impl")
RAY_DIST_NAMES = ("ray_dist.jar", "ray__dist.jar")


def find_ray_dist() -> pathlib.Path:
    ray_root = pathlib.Path(ray.__file__).resolve().parent
    matches = [
        candidate
        for candidate in ray_root.rglob("*.jar")
        if candidate.name in RAY_DIST_NAMES
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Ray dist jar under {ray_root}, found {matches}")
    return matches[0]


assert torch.cuda.is_available() or torch.version.cuda
assert torch.__version__.startswith("2.9.1")
assert Version(PIL.__version__) >= Version("12.2.0")
assert sglang
assert slime

ray_dist = find_ray_dist()
for artifact in LOG4J_ARTIFACTS:
    properties_name = f"META-INF/maven/org.apache.logging.log4j/{artifact}/pom.properties"
    with zipfile.ZipFile(ray_dist, "r") as jar:
        properties = jar.read(properties_name).decode("utf-8")
    assert "version=2.25.4" in properties, artifact

print("slime environment imports succeeded")
