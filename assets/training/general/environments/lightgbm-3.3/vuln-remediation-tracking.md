# Vulnerability remediation tracking

Image: `public/azureml/curated/lightgbm-3.3:83`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1`

Base SBOM status: `vcm image sbom download` found no SBOM artifact for the base image. The failed download attempt is recorded by the absence of `base-sbom.json`; package classification below uses the supplied VCM finding versions, Dockerfile `FROM` lineage, and an ACR apt-policy diagnostic against `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260707.v1`.

Validation status: `vcm image vulnerabilities evaluate --registry gatestacr.azurecr.io --repository public/azureml/curated/lightgbm-3.3 --tag test-fix` completed compliant with 0 non-compliant findings. The final image SBOM is saved as `sbom.json` for manual verification.

| Vulnerability | CVE(s) | Package(s) | Source | Base already covered? | File(s) changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- | --- |
| 6035947 / USN-8495-1 | CVE-2026-58055 | `libnghttp2-14` | Base-image Ubuntu package | Yes; base has `1.59.0-1ubuntu0.4` | `context/Dockerfile` comment cleanup only | `1.59.0-1ubuntu0.4` |
| 6035933 / USN-8487-1 | CVE-2026-8286, CVE-2026-8458, CVE-2026-8924, CVE-2026-8925, CVE-2026-8927, CVE-2026-9547 | `curl`, `libcurl4t64`, `libcurl3t64-gnutls` | Base-image Ubuntu packages | Yes; base has `8.5.0-2ubuntu10.10`, apt candidate is `8.5.0-2ubuntu10.11` | `context/Dockerfile` comment cleanup only | Final SBOM has `8.5.0-2ubuntu10.11` after `apt-get upgrade` |
| 6035919 / USN-8480-1 | CVE-2026-11822, CVE-2026-11824 | `libsqlite3-0` | Base-image Ubuntu package | Yes; base has `3.45.1-1ubuntu2.6` | `context/Dockerfile` comment cleanup only | `3.45.1-1ubuntu2.6` |
| 6035912 / USN-8477-1 | CVE-2026-5704 | `tar` | Base-image Ubuntu package | Yes; base has `1.35+dfsg-3ubuntu0.2` | `context/Dockerfile` comment cleanup only | `1.35+dfsg-3ubuntu0.2` |

Existing pins and overrides audited: current supplied VCM findings only identify Ubuntu OS packages above. Existing Python pins in `context/Dockerfile` and `context/conda_dependencies.yaml` are retained because the base SBOM was unavailable and there is no current VCM evidence that those prior CVE overrides are stale.
