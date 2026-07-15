# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:latest` (`sha256:d2675e1163fc7a9e69b870c0644508b56d88cebf99fdf0614ed7170c939db88a`)

Base SBOM status: `vcm image sbom download` found no SBOM artifact directly on MCR, and `vcm image sbom generate` cannot attach to MCR because the command only supports Azure Container Registry login server suffixes. To keep a comparable base SBOM for review, the same base-image digest was imported to `gatestacr.azurecr.io/temp/aoai-base-openmpi5-ubuntu24.04:latest`, then `base-sbom.json` and `base-vulnerabilities.json` were generated from that copy.

| Vulnerability | Source classification | Base image coverage | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- |
| 6035897 / USN-8458-1 / nginx | Inherited from the base image; the old Dockerfile also overrode nginx with stale exact pins. | Covered by base image at `1.24.0-2ubuntu7.13`. | `context/Dockerfile` | Removed stale exact nginx pins; no new nginx override. |
| 6035908 / USN-8456-1 / libxml2 | Inherited from the base image. | Covered by base image at `2.9.14+dfsg-1.3ubuntu3.8`. | None | No image-layer override. |
| 6035912 / USN-8477-1 / tar | Inherited from the base image. | Covered by base image at `1.35+dfsg-3ubuntu0.2`. | None | No image-layer override. |
| 6035919 / USN-8480-1 / SQLite | Inherited from the base image. | Covered by base image at `3.45.1-1ubuntu2.6`. | None | No image-layer override. |
| 6035933 / USN-8487-1 / curl | Inherited from the base image. | Partially covered by base image at `8.5.0-2ubuntu10.10`, but not the newer `8.5.0-2ubuntu10.11` required by USN-8525-1. | `context/Dockerfile` | Upgraded `curl`, `libcurl4t64`, `libcurl3t64-gnutls` to the current repo versions, at least `8.5.0-2ubuntu10.11`. |
| 6035947 / USN-8495-1 / nghttp2 | Inherited from the base image. | Covered by base image at `1.59.0-1ubuntu0.4`. | None | No image-layer override. |
| 6035952 / USN-8512-1 / gzip | Inherited from the base image and still vulnerable there. | Not covered by base image; base has `1.12-1ubuntu3.1`. | `context/Dockerfile` | Upgraded `gzip` to the current repo version, at least `1.12-1ubuntu3.2`. |
| 6035957 / USN-8510-1 / tar | Inherited from the base image. | Covered by base image at `1.35+dfsg-3ubuntu0.2`. | None | No image-layer override. |
| 6035961 / USN-8509-1 / Python 3.12 | Inherited from the base image. | Covered by base image at `3.12.3-1ubuntu0.15`. | None | No image-layer override. |
| 6035978 / USN-8525-1 / curl | Inherited from the base image and still vulnerable there. | Not covered by base image; base has `8.5.0-2ubuntu10.10`. | `context/Dockerfile` | Upgraded `curl`, `libcurl4t64`, `libcurl3t64-gnutls` to the current repo versions, at least `8.5.0-2ubuntu10.11`. |

Existing pins reviewed:

| Pin | Decision | Reason |
| --- | --- | --- |
| `libgnutls30t64` | Removed | Base image already carries the current patched version and the current scan does not flag it. |
| `openssl=3.0.13-0ubuntu3.11`, `libssl3t64=3.0.13-0ubuntu3.11` | Removed | Base image already carries the current patched versions and the current scan does not flag them. |
| `cryptography==48.0.1` | Kept | Existing Python transitive override for `azureml-mlflow`/`msal`/`azure-identity`; not part of the current requested vulnerability set. |
| `idna>=3.15` | Kept | Existing Python transitive override for `requests`/`azure-core`/`msal` and `httpx`/`openai`; not part of the current requested vulnerability set. |
| `urllib3>=2.7.0` | Kept | Existing Python transitive override for `requests`; not part of the current requested vulnerability set. |
