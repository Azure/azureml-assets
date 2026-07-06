# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Patch Ray's vendored Jackson Databind classes in its distribution jar."""

import hashlib
import io
import os
import pathlib
import shutil
import tempfile
import urllib.request
import zipfile

import ray


RAY_DIST_NAMES = ("ray_dist.jar", "ray__dist.jar")
ARTIFACT = "jackson-databind"
GROUP_PATH = "com/fasterxml/jackson/core"
MAVEN_METADATA_PREFIX = "META-INF/maven/com.fasterxml.jackson.core/jackson-databind/"
JACKSON_DATABIND_PREFIXES = (
    "com/fasterxml/jackson/databind/",
    "META-INF/services/com.fasterxml.jackson.databind.",
    MAVEN_METADATA_PREFIX,
)


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


def download_maven_artifacts(version: str) -> tuple[bytes, bytes]:
    """Download Jackson Databind jar and pom, verifying checksums."""
    base_url = f"https://repo1.maven.org/maven2/{GROUP_PATH}/{ARTIFACT}/{version}"
    jar = download_verified(
        f"{base_url}/{ARTIFACT}-{version}.jar",
        os.environ["JACKSON_DATABIND_JAR_SHA1"],
    )
    pom = download_verified(
        f"{base_url}/{ARTIFACT}-{version}.pom",
        os.environ["JACKSON_DATABIND_POM_SHA1"],
    )
    return jar, pom


def download_verified(url: str, expected_sha1: str) -> bytes:
    """Download a URL and verify its SHA1."""
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    actual_sha1 = hashlib.sha1(payload).hexdigest()
    if actual_sha1 != expected_sha1:
        raise RuntimeError(f"{url} checksum mismatch: {actual_sha1} != {expected_sha1}")
    return payload


def is_jackson_databind_entry(filename: str) -> bool:
    """Return whether a jar entry belongs to Jackson Databind."""
    return filename.startswith(JACKSON_DATABIND_PREFIXES) or any(
        fragment in filename
        for fragment in (
            "/com/fasterxml/jackson/databind/",
            "/META-INF/maven/com.fasterxml.jackson.core/jackson-databind/",
        )
    )


def copy_zip_info(source_info: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    """Copy zip metadata for a replacement entry."""
    target_info = zipfile.ZipInfo(filename, source_info.date_time)
    target_info.comment = source_info.comment
    target_info.extra = source_info.extra
    target_info.internal_attr = source_info.internal_attr
    target_info.external_attr = source_info.external_attr
    target_info.compress_type = source_info.compress_type
    target_info.create_system = source_info.create_system
    return target_info


def build_replacement_entries(
    jar_payload: bytes,
    pom_payload: bytes,
) -> dict[str, tuple[zipfile.ZipInfo, bytes]]:
    """Build replacement entries from the official Jackson Databind jar."""
    replacements = {}
    with zipfile.ZipFile(io.BytesIO(jar_payload), "r") as jar:
        for source_info in jar.infolist():
            if source_info.filename.endswith("/"):
                continue
            if not is_jackson_databind_entry(source_info.filename):
                continue
            replacements[source_info.filename] = (
                copy_zip_info(source_info, source_info.filename),
                jar.read(source_info.filename),
            )

    pom_name = f"{MAVEN_METADATA_PREFIX}pom.xml"
    if pom_name in replacements:
        replacements[pom_name] = (replacements[pom_name][0], pom_payload)
    else:
        replacements[pom_name] = (zipfile.ZipInfo(pom_name), pom_payload)
    return replacements


def patch_ray_dist(
    ray_dist: pathlib.Path,
    version: str,
    replacements: dict[str, tuple[zipfile.ZipInfo, bytes]],
) -> None:
    """Overlay Jackson Databind replacement entries into Ray's distribution jar."""
    fd, patched_name = tempfile.mkstemp(suffix=".jar")
    os.close(fd)
    patched_path = pathlib.Path(patched_name)
    written_replacements = set()
    removed_old_entries = []

    try:
        with zipfile.ZipFile(ray_dist, "r") as source, zipfile.ZipFile(
            patched_path, "w"
        ) as target:
            for source_info in source.infolist():
                if source_info.filename in replacements:
                    target_info, payload = replacements[source_info.filename]
                    target.writestr(target_info, payload)
                    written_replacements.add(source_info.filename)
                elif is_jackson_databind_entry(source_info.filename):
                    removed_old_entries.append(source_info.filename)
                else:
                    target.writestr(source_info, source.read(source_info.filename))

            for filename, (target_info, payload) in replacements.items():
                if filename not in written_replacements:
                    target.writestr(target_info, payload)
                    written_replacements.add(filename)

        if not written_replacements:
            raise RuntimeError(f"No Jackson Databind replacement entries were written to {ray_dist}")

        shutil.move(str(patched_path), ray_dist)
    finally:
        patched_path.unlink(missing_ok=True)

    print(
        f"Patched {ray_dist}; wrote {len(written_replacements)} Jackson Databind "
        f"entries and removed {len(removed_old_entries)} old-only entries"
    )


def validate_patch(ray_dist: pathlib.Path, version: str) -> None:
    """Validate that Ray's jar reports the expected Jackson Databind version."""
    properties_name = f"{MAVEN_METADATA_PREFIX}pom.properties"
    with zipfile.ZipFile(ray_dist, "r") as jar:
        properties = jar.read(properties_name).decode("utf-8")
    expected = f"version={version}"
    if expected not in properties:
        raise RuntimeError(f"{properties_name} does not contain {expected}")


def main() -> None:
    """Patch Ray's vendored Jackson Databind contents."""
    version = os.environ["JACKSON_DATABIND_VERSION"]
    ray_dist = find_ray_dist()
    jar_payload, pom_payload = download_maven_artifacts(version)
    replacements = build_replacement_entries(jar_payload, pom_payload)
    patch_ray_dist(ray_dist, version, replacements)
    validate_patch(ray_dist, version)
    print(f"Patched Ray dist jar with {ARTIFACT}-{version}.jar")


if __name__ == "__main__":
    main()
