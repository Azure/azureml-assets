# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
LOG4J_ARTIFACTS = {
    "log4j-api": "LOG4J_API_SHA1",
    "log4j-core": "LOG4J_CORE_SHA1",
    "log4j-slf4j-impl": "LOG4J_SLF4J_IMPL_SHA1",
}
LOG4J_PREFIXES = (
    "META-INF/maven/org.apache.logging.log4j/",
    "META-INF/org/apache/logging/log4j/",
    "META-INF/services/org.apache.logging.log4j",
    "org/apache/logging/log4j/",
    "org/apache/logging/slf4j/",
    "org/slf4j/impl/",
)
SKIP_REPLACEMENT_PREFIXES = (
    "META-INF/MANIFEST.MF",
    "META-INF/LICENSE",
    "META-INF/NOTICE",
    "META-INF/DEPENDENCIES",
)


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


def download_log4j_jars(version: str) -> dict[str, bytes]:
    jars = {}
    for artifact, checksum_env in LOG4J_ARTIFACTS.items():
        expected_sha1 = os.environ[checksum_env]
        url = (
            "https://repo1.maven.org/maven2/org/apache/logging/log4j/"
            f"{artifact}/{version}/{artifact}-{version}.jar"
        )
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
        actual_sha1 = hashlib.sha1(payload).hexdigest()
        if actual_sha1 != expected_sha1:
            raise RuntimeError(
                f"{artifact} checksum mismatch: {actual_sha1} != {expected_sha1}"
            )
        jars[artifact] = payload
    return jars


def is_log4j_owned_entry(filename: str) -> bool:
    return filename.startswith(LOG4J_PREFIXES) or any(
        path_fragment in filename
        for path_fragment in (
            "/org/apache/logging/log4j/",
            "/org/apache/logging/slf4j/",
            "/org/slf4j/impl/",
        )
    )


def copy_zip_info(source_info: zipfile.ZipInfo, filename: str) -> zipfile.ZipInfo:
    target_info = zipfile.ZipInfo(filename, source_info.date_time)
    target_info.comment = source_info.comment
    target_info.extra = source_info.extra
    target_info.internal_attr = source_info.internal_attr
    target_info.external_attr = source_info.external_attr
    target_info.compress_type = source_info.compress_type
    target_info.create_system = source_info.create_system
    return target_info


def build_replacement_entries(jars: dict[str, bytes]) -> dict[str, tuple[zipfile.ZipInfo, bytes]]:
    replacements = {}
    for artifact, payload in jars.items():
        with zipfile.ZipFile(io.BytesIO(payload), "r") as jar:
            for source_info in jar.infolist():
                if source_info.filename.endswith("/"):
                    continue
                if source_info.filename.startswith(SKIP_REPLACEMENT_PREFIXES):
                    continue
                if not is_log4j_owned_entry(source_info.filename):
                    continue
                if source_info.filename in replacements:
                    raise RuntimeError(
                        f"Duplicate replacement entry {source_info.filename} from {artifact}"
                    )
                replacements[source_info.filename] = (
                    copy_zip_info(source_info, source_info.filename),
                    jar.read(source_info.filename),
                )
    return replacements


def patch_ray_dist(
    ray_dist: pathlib.Path,
    version: str,
    replacements: dict[str, tuple[zipfile.ZipInfo, bytes]],
) -> None:
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
                elif is_log4j_owned_entry(source_info.filename):
                    removed_old_entries.append(source_info.filename)
                else:
                    target.writestr(source_info, source.read(source_info.filename))

            for filename, (target_info, payload) in replacements.items():
                if filename not in written_replacements:
                    target.writestr(target_info, payload)
                    written_replacements.add(filename)

        if not written_replacements:
            raise RuntimeError(f"No Log4j replacement entries were written to {ray_dist}")

        shutil.move(str(patched_path), ray_dist)
    finally:
        patched_path.unlink(missing_ok=True)

    print(
        f"Patched {ray_dist}; wrote {len(written_replacements)} Log4j entries "
        f"and removed {len(removed_old_entries)} old-only entries"
    )


def validate_patch(ray_dist: pathlib.Path, version: str) -> None:
    with zipfile.ZipFile(ray_dist, "r") as jar:
        for artifact in LOG4J_ARTIFACTS:
            properties_name = (
                f"META-INF/maven/org.apache.logging.log4j/{artifact}/pom.properties"
            )
            properties = jar.read(properties_name).decode("utf-8")
            expected = f"version={version}"
            if expected not in properties:
                raise RuntimeError(f"{properties_name} does not contain {expected}")


def main() -> None:
    version = os.environ["LOG4J_VERSION"]
    ray_dist = find_ray_dist()
    jars = download_log4j_jars(version)
    replacements = build_replacement_entries(jars)
    patch_ray_dist(ray_dist, version, replacements)
    validate_patch(ray_dist, version)
    print(
        "Patched Ray dist jar with "
        + ", ".join(f"{artifact}-{version}.jar" for artifact in LOG4J_ARTIFACTS)
    )


if __name__ == "__main__":
    main()
