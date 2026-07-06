# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Smoke-test the slime curated environment image."""

import importlib.util
import pathlib
import zipfile
import cryptography
from packaging.version import Version
import PIL
import ray
import sglang
import slime
import torch
import transformer_engine.pytorch as te


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


def assert_slime_resolves_under_expected_root() -> None:
    """Verify slime resolves to the readable editable source tree."""
    spec = importlib.util.find_spec("slime")
    assert spec is not None, "importlib cannot find slime"

    paths: list[pathlib.Path] = []
    slime_file = getattr(slime, "__file__", None)
    if slime_file:
        paths.append(pathlib.Path(slime_file).resolve())
    if spec.origin and spec.origin != "namespace":
        paths.append(pathlib.Path(spec.origin).resolve())
    for location in spec.submodule_search_locations or ():
        paths.append(pathlib.Path(location).resolve())
    for location in getattr(slime, "__path__", ()):
        paths.append(pathlib.Path(location).resolve())

    assert paths, "slime imported but no source paths were discoverable"
    assert any(
        path == EXPECTED_SLIME_ROOT or EXPECTED_SLIME_ROOT in path.parents
        for path in paths
    ), f"slime resolved at {paths}, expected under {EXPECTED_SLIME_ROOT}"


def assert_world_accessible(path: pathlib.Path) -> None:
    """Verify a file/tree is readable by the non-root AML job user."""
    assert path.exists(), f"missing {path}"
    mode = path.stat().st_mode & 0o777
    assert mode & 0o004, f"{path} not world-readable (mode={oct(mode)})"
    if path.is_dir():
        assert mode & 0o001, f"{path} not world-traversable (mode={oct(mode)})"


assert torch.cuda.is_available() or torch.version.cuda
assert torch.__version__.startswith("2.10.0")
assert Version(PIL.__version__) >= Version("12.2.0")
assert Version(cryptography.__version__) >= Version("48.0.1")
assert Version(sglang.__version__) >= Version("0.5.11")
assert sglang
assert slime
assert te

# AML/Singularity jobs run as uid 9000 (aiscuser); /root is mode 700 so
# slime must be editable-installed from a non-/root location. Pin the
# expected location and verify world read+traverse on key files.
assert_slime_resolves_under_expected_root()
for path in (
    EXPECTED_SLIME_ROOT,
    EXPECTED_SLIME_ROOT / "train.py",
    EXPECTED_SLIME_ROOT / "slime",
    pathlib.Path("/opt/Megatron-LM"),
):
    assert_world_accessible(path)

slime_init = EXPECTED_SLIME_ROOT / "slime" / "__init__.py"
if slime_init.exists():
    assert_world_accessible(slime_init)

initialize_py = EXPECTED_SLIME_ROOT / "slime" / "backends" / "megatron_utils" / "initialize.py"
model_provider_py = EXPECTED_SLIME_ROOT / "slime" / "backends" / "megatron_utils" / "model_provider.py"
assert "Megatron does not support numpy 2.x" not in initialize_py.read_text(encoding="utf-8")
assert "provider.attention_backend = args.attention_backend" in model_provider_py.read_text(
    encoding="utf-8"
)

ray_dist = find_ray_dist()
for artifact in LOG4J_ARTIFACTS:
    properties_name = f"META-INF/maven/org.apache.logging.log4j/{artifact}/pom.properties"
    with zipfile.ZipFile(ray_dist, "r") as jar:
        properties = jar.read(properties_name).decode("utf-8")
    assert "version=2.25.4" in properties, artifact

print("slime environment imports succeeded")
