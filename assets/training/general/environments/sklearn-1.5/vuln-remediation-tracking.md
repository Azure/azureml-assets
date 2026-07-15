# Vulnerability remediation tracking

Image: `public/azureml/curated/sklearn-1.5:49`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1`

Base SBOM status: `vcm image sbom download` found no SBOM artifact for the base image. The failed download attempt is recorded by the absence of `base-sbom.json`; package classification below uses the Dockerfile `FROM` lineage plus ACR validation of the resolved base image package versions.

| Vulnerability | CVE(s) | Package(s) | Source | Base already covered? | File(s) changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- | --- |
| 6035912 / USN-8477-1 | CVE-2026-5704 | `tar` | Base image | Yes; resolved base has `1.35+dfsg-3ubuntu0.2` | None | Base version `1.35+dfsg-3ubuntu0.2` |
| 6035919 / USN-8480-1 | CVE-2026-11822, CVE-2026-11824 | `libsqlite3-0` | Base image | Yes; resolved base has `3.45.1-1ubuntu2.6` | None | Base version `3.45.1-1ubuntu2.6` |
| 6035933 / USN-8487-1 | CVE-2026-8286, CVE-2026-8458, CVE-2026-8924, CVE-2026-8925, CVE-2026-8927, CVE-2026-9547 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base image | Yes; resolved base has `8.5.0-2ubuntu10.10` and image-layer `apt-get upgrade` advances to candidate `8.5.0-2ubuntu10.11` | None | Base version `8.5.0-2ubuntu10.10`; upgrade candidate `8.5.0-2ubuntu10.11` |
| 6035947 / USN-8495-1 | CVE-2026-58055 | `libnghttp2-14` | Base image | Yes; resolved base has `1.59.0-1ubuntu0.4` | None | Base version `1.59.0-1ubuntu0.4` |
| 5015223 / GHSA-wf93-45jw-7689 | CVE-2026-8643 | `pip` | This image's conda environment | No; direct image pin installed `26.1.1` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `26.1.2` |

Existing pins and overrides audited: the stale direct `pip=26.1.1` security pin was updated to `26.1.2` after the current image scan reported CVE-2026-8643. Other existing Python pins in `context/Dockerfile` and `context/conda_dependencies.yaml` are retained because there is no current VCM evidence that those prior CVE overrides are stale.
