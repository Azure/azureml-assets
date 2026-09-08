# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn:48`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}`, resolved for verification to `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`.

Base SBOM/finding files used for manual verification:

- `base-sbom.json`
- `base-vulnerabilities.json`

Remediated image SBOM/finding files used for manual verification:

- `sbom.json`
- `vulnerabilities.json`

| Finding | Package(s) | Source classification | Base image status | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873105 / USN-8670-1 | `curl`, `libcurl3t64-gnutls`, `libcurl4t64` | Base image | Already covered in the base SBOM at `8.5.0-2ubuntu10.13`; no image-layer override needed. | None | `8.5.0-2ubuntu10.13` |
| 6873052 / USN-8651-1 | `curl`, `libcurl3t64-gnutls`, `libcurl4t64` | Base image | Already covered in the base SBOM at `8.5.0-2ubuntu10.13`; no image-layer override needed. | None | `8.5.0-2ubuntu10.13` |
| 6873087 / USN-8678-1 | `libssl3t64`, `openssl` | Base image | Already covered in the base SBOM at `3.0.13-0ubuntu3.15`; no image-layer override needed. | None | `3.0.13-0ubuntu3.15` |
| 6873114 / USN-8684-1 | `perl`, `libperl5.38t64`, `perl-modules-5.38`, `perl-base` | Base image | Already covered in the base SBOM at `5.38.2-3.2ubuntu0.4`; no image-layer override needed. | None | `5.38.2-3.2ubuntu0.4` |
| 6873121 / USN-8687-1 | `libp11-kit0` | Base image | Already covered in the base SBOM at `0.25.3-4ubuntu2.2`; no image-layer override needed. | None | `0.25.3-4ubuntu2.2` |
| 6873124 / USN-8691-1 | `libattr1` | Base image | Already covered in the base SBOM at `1:2.5.2-1ubuntu0.1`; no image-layer override needed. | None | `1:2.5.2-1ubuntu0.1` |
| 6873125 / USN-8699-1 | `libssh-4` | Base image | Already covered in the base SBOM at `0.10.6-2ubuntu0.5`; no image-layer override needed. | None | `0.10.6-2ubuntu0.5` |
| 6873134 / USN-8692-1 | `diffutils` | Base image | Already covered in the base SBOM at `1:3.10-1ubuntu0.1`; no image-layer override needed. | None | `1:3.10-1ubuntu0.1` |
| 6873135 / USN-8697-1 | `coreutils` | Base image | Already covered in the base SBOM at `9.4-3ubuntu6.3`; no image-layer override needed. | None | `9.4-3ubuntu6.3` |
| 6873141 / USN-8706-1 | `zlib1g` | Base image | Already covered in the base SBOM at `1:1.3.dfsg-3.1ubuntu2.2`; no image-layer override needed. | None | `1:1.3.dfsg-3.1ubuntu2.2` |
| 6873148 / USN-8711-1 | `libgcrypt20` | Base image | Base SBOM contains vulnerable `1.10.3-2ubuntu0.1`; this image's apt-upgrade layer upgrades it. | `context\Dockerfile` | `1.10.3-2ubuntu0.2` |

Existing Dockerfile pins and overrides were audited against the current image and base evidence. No stale OS-package pin was added for packages already fixed by the base image. The remaining Python package security overrides are unrelated to the Ubuntu package findings above and still document active transitive dependency or compatibility constraints.
