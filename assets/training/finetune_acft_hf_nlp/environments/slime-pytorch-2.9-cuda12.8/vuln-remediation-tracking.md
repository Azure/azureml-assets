# Vulnerability remediation tracking

Image: `public/azureml/curated/slime-pytorch-2.9-cuda12.8:7`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04:latest`

Base SBOM status: `vcm image sbom download` found no SBOM artifact for the base image digest, so base-image package comparison is recorded as unavailable. The vulnerable packages below are introduced or overridden by this image: PyTorch is installed directly in the Dockerfile, and Ray is installed from `context/requirements.txt`.

Validation: built `vulnscan1779267129n13.azurecr.io/public/azureml/curated/slime-pytorch-2.9-cuda12.8:test-fix`, generated/downloaded `sbom.json`, and `vcm image vulnerabilities evaluate --sbom sbom.json` reported `compliant` with 0 non-compliant findings.

| Finding | Package | Source | Base already covered | File(s) changed | Patched version |
| --- | --- | --- | --- | --- | --- |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image direct Dockerfile install | No base SBOM available; this image overrides torch | `context/Dockerfile`, `context/smoke_test.py` | `2.10.0+cu128` |
| 5014608 / GHSA-hgj6-7826-r7m5 | `com.fasterxml.jackson.core:jackson-databind` in Ray `ray__dist.jar` | This image installs `ray[default]==2.55.1` via `context/requirements.txt` | No base SBOM available; Ray is added by this image | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.19.4` |
| 5014607 / GHSA-5jmj-h7xm-6q6v | `com.fasterxml.jackson.core:jackson-databind` in Ray `ray__dist.jar` | This image installs `ray[default]==2.55.1` via `context/requirements.txt` | No base SBOM available; Ray is added by this image | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.19.4` |
| 5014605 / GHSA-j3rv-43j4-c7qm | `com.fasterxml.jackson.core:jackson-databind` in Ray `ray__dist.jar` | This image installs `ray[default]==2.55.1` via `context/requirements.txt` | No base SBOM available; Ray is added by this image | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.19.4` |
| 5014600 / GHSA-rmj7-2vxq-3g9f | `com.fasterxml.jackson.core:jackson-databind` in Ray `ray__dist.jar` | This image installs `ray[default]==2.55.1` via `context/requirements.txt` | No base SBOM available; Ray is added by this image | `context/Dockerfile`, `context/patch_ray_jackson_databind.py` | `2.19.4` |

Existing pins/overrides audited:

| Package or override | Status |
| --- | --- |
| `aiohttp>=3.14.0`, `starlette>=1.0.1`, `idna>=3.15`, `pyarrow>=23.0.1` | Kept; these address existing scanner findings documented in Dockerfile comments. |
| `cryptography==48.0.1` | Kept; existing transitive base-image vulnerability pin documented in Dockerfile. |
| Ray vendored `idna` and `aiohttp` patch | Kept; required because Ray bundles third-party package metadata outside main site-packages. |
| Ray vendored Log4j patch | Kept; required because Ray bundles Maven metadata and classes inside `ray_dist.jar`. |
