# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Smoke-test the slime curated environment image."""

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
EXPECTED_SLIME_ROOT = pathlib.Path("/opt/slime")


def find_ray_dist() -> pathlib.Path:
    """Find the Ray distribution fat jar."""
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

# AML/Singularity jobs run as uid 9000 (aiscuser); /root is mode 700 so
# slime must be editable-installed from a non-/root location. Pin the
# expected location and verify world read+traverse on key files.
slime_path = pathlib.Path(slime.__file__).resolve()
assert EXPECTED_SLIME_ROOT in slime_path.parents, (
    f"slime resolved at {slime_path}, expected under {EXPECTED_SLIME_ROOT}"
)
for path in (
    EXPECTED_SLIME_ROOT,
    EXPECTED_SLIME_ROOT / "train.py",
    EXPECTED_SLIME_ROOT / "slime" / "__init__.py",
    pathlib.Path("/opt/Megatron-LM"),
):
    assert path.exists(), f"missing {path}"
    mode = path.stat().st_mode & 0o777
    assert mode & 0o004, f"{path} not world-readable (mode={oct(mode)})"
    if path.is_dir():
        assert mode & 0o001, f"{path} not world-traversable (mode={oct(mode)})"

ray_dist = find_ray_dist()
for artifact in LOG4J_ARTIFACTS:
    properties_name = f"META-INF/maven/org.apache.logging.log4j/{artifact}/pom.properties"
    with zipfile.ZipFile(ray_dist, "r") as jar:
        properties = jar.read(properties_name).decode("utf-8")
    assert "version=2.25.4" in properties, artifact

print("slime environment imports succeeded")
