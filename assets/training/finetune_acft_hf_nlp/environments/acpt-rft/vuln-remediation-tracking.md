# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-rft-training:20`

Base image resolved from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

Base SBOM note: `vcm image sbom download` against `mcr.microsoft.com` found no SBOM referrer for the resolved base digest. `base-sbom.json` and `base-vulnerabilities.json` in this directory are copied from the existing same-digest generated ACPT base evidence for manual verification.

| VCM ID | Package | Source | Base already covered? | File(s) changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014607 | `com.fasterxml.jackson.core:jackson-databind` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.21.4` |
| 5014608 | `com.fasterxml.jackson.core:jackson-databind` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.21.4` |
| 5014600 | `com.fasterxml.jackson.core:jackson-databind` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.21.4` |
| 5014605 | `com.fasterxml.jackson.core:jackson-databind` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.21.4` |
| 5010908 | `org.apache.logging.log4j:log4j-core` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.25.4` |
| 5010910 | `org.apache.logging.log4j:log4j-core` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.25.4` |
| 5010747 | `org.apache.logging.log4j:log4j-core` in Ray jar | This image, via `ray[default]` | No | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.25.4` |
| 5013672 | `vllm` | This image, direct runtime install | N/A | `context/Dockerfile` | `0.22.0` |
| 5013910 | `vllm` | This image, direct runtime install | N/A | `context/Dockerfile` | `0.22.0` |
| 5013924 | `vllm` | This image, direct runtime install | N/A | `context/Dockerfile` | `0.22.0` |
| 5013832 | `cryptography` | This image package resolution can leave older copies in base and ptca envs | No | `context/Dockerfile` | `>=48.0.1` |
| 5014292 | `pydantic-settings` | This image, transitive dependency of `fastmcp` | No | `context/Dockerfile` | `>=2.14.2` |
| 5014304 | `msgpack` | This image/base Python env package resolution | No | `context/Dockerfile` | `>=1.2.1` |
| 5015223 | `pip` | This image/base conda metadata and ptca env metadata | No | `context/Dockerfile` | `>=26.1.2`; stale `pip-26.1.1*.json` metadata removed |
| 6035947 | `libnghttp2-14` | Base image OS package | Yes | None | Base has `1.43.0-1ubuntu0.4`; Dockerfile keeps `apt-get upgrade` |
| 6035908 | `libxml2` | Base image OS package | Yes | None | Base has `2.9.13+dfsg-1ubuntu0.12`; Dockerfile keeps `apt-get upgrade` |
| 6035912 | `tar` | Base image OS package | Yes | None | Base has `1.34+dfsg-1ubuntu0.1.22.04.4`; Dockerfile keeps `apt-get upgrade` |
| 6035957 | `tar` | Base image OS package | Yes | None | Base has `1.34+dfsg-1ubuntu0.1.22.04.4`; Dockerfile keeps `apt-get upgrade` |
| 6035919 | `libsqlite3-0` | Base image OS package | Yes | None | Base has `3.37.2-2ubuntu0.6`; Dockerfile keeps `apt-get upgrade` |
| 6035933 | `curl`, `libcurl3-gnutls`, `libcurl4` | Base image OS package | Yes | None | Base has `7.81.0-1ubuntu1.25`; Dockerfile keeps `apt-get upgrade` |
| 6035952 | `gzip` | Base image OS package | Yes | None | Base has `1.10-4ubuntu4.2`; Dockerfile keeps `apt-get upgrade` |
| 6035961 | `python3.10`, `libpython3.10-*` | Base image OS package | Yes | None | Base has `3.10.12-1~22.04.16`; Dockerfile keeps `apt-get upgrade` |

