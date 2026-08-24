# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Patch Ray's vendored HttpCore5 artifacts in its distribution jar."""

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
HTTPCORE5_PREFIXES = (
    "META-INF/maven/org.apache.httpcomponents.core5/httpcore5/",
    "org/apache/hc/core5/annotation/",
    "org/apache/hc/core5/concurrent/",
    "org/apache/hc/core5/function/",
    "org/apache/hc/core5/http/",
    "org/apache/hc/core5/io/",
    "org/apache/hc/core5/net/",
    "org/apache/hc/core5/pool/",
    "org/apache/hc/core5/reactor/",
    "org/apache/hc/core5/ssl/",
    "org/apache/hc/core5/testing/",
    "org/apache/hc/core5/util/",
)


def find_ray_dist() -> pathlib.Path:
    """Find Ray's vendored distribution jar."""
    ray_root = pathlib.Path(ray.__file__).resolve().parent
    matches = [
        candidate
        for candidate in ray_root.rglob("*.jar")
        if candidate.name in RAY_DIST_NAMES
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one Ray dist jar under {ray_root}, found {matches}")
    return matches[0]


def download_verified(url: str, expected_sha1: str) -> bytes:
    """Download a URL and verify its SHA-1 checksum."""
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    actual_sha1 = hashlib.sha1(payload).hexdigest()
    if actual_sha1 != expected_sha1:
        raise RuntimeError(f"{url} checksum mismatch: {actual_sha1} != {expected_sha1}")
    return payload


def download_maven_artifacts(version: str) -> tuple[bytes, bytes]:
    """Download the Maven jar and pom artifacts."""
    base_url = (
        "https://repo.maven.apache.org/maven2/org/apache/httpcomponents/core5/"
        f"httpcore5/{version}"
    )
    jar = download_verified(
        f"{base_url}/httpcore5-{version}.jar",
        os.environ["HTTPCORE5_JAR_SHA1"],
    )
    pom = download_verified(
        f"{base_url}/httpcore5-{version}.pom",
        os.environ["HTTPCORE5_POM_SHA1"],
    )
    return jar, pom


def is_httpcore5_entry(filename: str) -> bool:
    """Return whether a jar entry belongs to the HttpCore5 Maven artifact."""
    return filename.startswith(HTTPCORE5_PREFIXES)


def copy_zip_info(source_info: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    """Copy zip entry metadata while replacing the filename."""
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
    """Build replacement entries from downloaded Maven artifacts."""
    replacements = {}
    with zipfile.ZipFile(io.BytesIO(jar_payload), "r") as jar:
        for source_info in jar.infolist():
            if source_info.filename.endswith("/"):
                continue
            if not is_httpcore5_entry(source_info.filename):
                continue
            replacements[source_info.filename] = (
                copy_zip_info(source_info, source_info.filename),
                jar.read(source_info.filename),
            )

    pom_name = "META-INF/maven/org.apache.httpcomponents.core5/httpcore5/pom.xml"
    replacements[pom_name] = (
        replacements[pom_name][0] if pom_name in replacements else zipfile.ZipInfo(pom_name),
        pom_payload,
    )
    return replacements


def patch_ray_dist(
    ray_dist: pathlib.Path,
    replacements: dict[str, tuple[zipfile.ZipInfo, bytes]],
) -> None:
    """Patch Ray's distribution jar with replacement HttpCore5 entries."""
    fd, patched_name = tempfile.mkstemp(suffix=".jar")
    os.close(fd)
    patched_path = pathlib.Path(patched_name)
    written_replacements = set()
    removed_old_entries = []

    try:
        with zipfile.ZipFile(ray_dist, "r") as source, zipfile.ZipFile(
            patched_path,
            "w",
        ) as target:
            for source_info in source.infolist():
                if source_info.filename in replacements:
                    target_info, payload = replacements[source_info.filename]
                    target.writestr(target_info, payload)
                    written_replacements.add(source_info.filename)
                elif is_httpcore5_entry(source_info.filename):
                    removed_old_entries.append(source_info.filename)
                else:
                    target.writestr(source_info, source.read(source_info.filename))

            for filename, (target_info, payload) in replacements.items():
                if filename not in written_replacements:
                    target.writestr(target_info, payload)
                    written_replacements.add(filename)

        if not written_replacements:
            raise RuntimeError(f"No HttpCore5 replacement entries were written to {ray_dist}")

        shutil.move(str(patched_path), ray_dist)
    finally:
        patched_path.unlink(missing_ok=True)

    print(
        f"Patched {ray_dist}; wrote {len(written_replacements)} HttpCore5 "
        f"entries and removed {len(removed_old_entries)} old-only entries",
    )


def validate_patch(ray_dist: pathlib.Path, version: str) -> None:
    """Validate that the patched jar reports the expected HttpCore5 version."""
    properties_name = "META-INF/maven/org.apache.httpcomponents.core5/httpcore5/pom.properties"
    with zipfile.ZipFile(ray_dist, "r") as jar:
        properties = jar.read(properties_name).decode("utf-8")
    expected = f"version={version}"
    if expected not in properties:
        raise RuntimeError(f"{properties_name} does not contain {expected}")


def main() -> None:
    """Patch Ray's vendored HttpCore5 dependency."""
    version = os.environ["HTTPCORE5_VERSION"]
    ray_dist = find_ray_dist()
    jar_payload, pom_payload = download_maven_artifacts(version)
    replacements = build_replacement_entries(jar_payload, pom_payload)
    patch_ray_dist(ray_dist, replacements)
    validate_patch(ray_dist, version)
    print(f"Patched Ray dist jar with httpcore5-{version}.jar")


if __name__ == "__main__":
    main()
