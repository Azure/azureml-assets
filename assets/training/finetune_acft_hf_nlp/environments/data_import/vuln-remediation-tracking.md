# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-hf-nlp-data-import:32`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1` (resolved from `{{latest-image-tag}}`). Direct MCR SBOM download failed twice with `ConnectionResetError(10054)`, so the base image was imported to `vulnscan1779267129n11.azurecr.io/temp/acft-hf-nlp-data-import-base:20260707.v1`; `base-sbom.json` and `base-vulnerabilities.json` were generated/downloaded from that ACR copy.

Scan artifacts are retained locally as `sbom.json` and `vulnerabilities.json`; base image artifacts are retained as `base-sbom.json` and `base-vulnerabilities.json`.

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035912 / USN-8477-1 | `tar` | Base-image OS package | Yes; base SBOM has `1.35+dfsg-3ubuntu0.2` | None | No local override |
| 6035919 / USN-8480-1 | `libsqlite3-0`, `sqlite3` | `libsqlite3-0` is a base-image OS package; `sqlite3` is installed by this image | `libsqlite3-0` covered by base at `3.45.1-1ubuntu2.6`; `sqlite3` is fixed in this layer | `context\Dockerfile` | `3.45.1-1ubuntu2.6` |
| 6035933 / USN-8487-1 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base-image OS packages | Yes for this older advisory; base SBOM has `8.5.0-2ubuntu10.10` | None | No local override for USN-8487-1 |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base-image OS package | Yes; base SBOM has `1.59.0-1ubuntu0.4` | None | No local override |
| 6035952 / USN-8512-1 | `gzip` | Base-image OS package | No; base SBOM has `1.12-1ubuntu3.1` | `context\Dockerfile` | `1.12-1ubuntu3.2` |
| 6035957 / USN-8510-1 | `tar` | Base-image OS package | Yes; base SBOM has `1.35+dfsg-3ubuntu0.2` | None | No local override |
| 6035961 / USN-8509-1 | `python3.12`, `python3.12-minimal`, `libpython3.12-minimal`, `libpython3.12-stdlib` | Base-image OS packages | Yes; base SBOM has `3.12.3-1ubuntu0.15` | None | No local override |
| 6035978 / USN-8525-1 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base-image OS packages | No; base SBOM has `8.5.0-2ubuntu10.10` | `context\Dockerfile` | `8.5.0-2ubuntu10.11` |

Existing Dockerfile pins and overrides were audited against the generated base SBOM and requested findings. The Python package overrides are unrelated to these Ubuntu OS package findings and remain in place because they protect packages installed into Python prefixes after the base image is built. Stale detailed comments for older OS findings were removed; the remaining OS upgrade list keeps inherited packages at the current noble-security level without adding transitive package pins.
