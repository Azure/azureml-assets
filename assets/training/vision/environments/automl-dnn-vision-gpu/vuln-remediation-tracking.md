# Vulnerability remediation tracking

Base image resolved from the Dockerfile template to `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202601.1`. `vcm image sbom download` found no SBOM artifact for that base image, so source classification below uses the affected install paths and Dockerfile/package ownership. Generated `base-sbom.json` and `sbom.json` are kept uncommitted for manual review when available.

| Vulnerability | Package(s) | Source | Base covered? | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035897 / USN-8458-1 | `nginx-light`, `nginx-common`, `libnginx-mod-http-geoip2`, `libnginx-mod-http-echo` | This image installs nginx for inference | No | `context/Dockerfile` | `1.18.0-6ubuntu14.16` |
| 6035908 / USN-8456-1 | `libxml2` | Base OS package inherited into this image | No | `context/Dockerfile` | `2.9.13+dfsg-1ubuntu0.12` |
| 6035912 / USN-8477-1 | `tar` | Base OS package inherited into this image | No | `context/Dockerfile` | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Base OS package inherited into this image | No | `context/Dockerfile` | `3.37.2-2ubuntu0.6` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Base OS package inherited into this image | No | `context/Dockerfile` | `7.81.0-1ubuntu1.25` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base OS package inherited into this image | No | `context/Dockerfile` | `1.43.0-1ubuntu0.4` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base conda Python 3.13 environment | No | `context/Dockerfile` | `>=2.14.2` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base conda Python 3.13 environment | No | `context/Dockerfile` | `>=1.2.1` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | This image active conda env and inherited ACPT `ptca` env | No | `context/Dockerfile` | `2.10.0` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image active conda env and inherited ACPT `ptca` env | No | `context/Dockerfile` | `2.10.0` |
