# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-vision-gpu:54`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}`, resolved for verification to `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`.

Base SBOM/finding files used for manual verification:

- `base-sbom.json`
- `base-vulnerabilities.json`

Remediated image SBOM/finding files used for manual verification:

- `sbom.json`
- `vulnerabilities.json`

Note: `vcm image sbom download` found no SBOM referrer for the base image in MCR during this remediation run, so `base-sbom.json` is the existing asset-local base SBOM for `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`.

| Finding | Package(s) | Source classification | Base image status | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873121 / USN-8687-1 | `libp11-kit0` | Base image | Already covered in the base SBOM at `0.25.3-4ubuntu2.2`; no new image-layer pin needed. | None | `0.25.3-4ubuntu2.2` |
| 6873124 / USN-8691-1 | `libattr1` | Base image | Already covered in the base SBOM at `1:2.5.2-1ubuntu0.1`; no new image-layer pin needed. | None | `1:2.5.2-1ubuntu0.1` |
| 6873125 / USN-8699-1 | `libssh-4` | Base image | Already covered in the base SBOM at `0.10.6-2ubuntu0.5`; no new image-layer pin needed. | None | `0.10.6-2ubuntu0.5` |
| 6873134 / USN-8692-1 | `diffutils` | Base image | Already covered in the base SBOM at `1:3.10-1ubuntu0.1`; no new image-layer pin needed. | None | `1:3.10-1ubuntu0.1` |
| 6873135 / USN-8697-1 | `coreutils` | Base image | Already covered in the base SBOM at `9.4-3ubuntu6.3`; no new image-layer pin needed. | None | `9.4-3ubuntu6.3` |
| 6873148 / USN-8711-1 | `libgcrypt20` | Base image | Base SBOM contains vulnerable `1.10.3-2ubuntu0.1`; this image's existing apt-upgrade layer upgrades it. | `context\Dockerfile` | `1.10.3-2ubuntu0.2` |
| 6873149 / USN-8710-1 | `libevent-core-2.1-7` | Base image package name for Ubuntu 22.04 variants | Not present in the Ubuntu 24.04 base or final SBOM for this image; no new image-layer pin needed for this build. | None | `2.1.12-stable-1ubuntu0.1` |
| 5017848 / GHSA-qwm4-qh6w-59xr | `pip` in `/opt/conda/envs/ptca` | Inherited PTCA conda environment when present | Final SBOM reports `pip` `26.2.1`; Dockerfile keeps the existing conda install override because VCM reads the `/opt/conda/envs/ptca/conda-meta` record. | `context\Dockerfile` | `26.2.0` or newer |

Existing Dockerfile pins and overrides were audited against the current image and base evidence. No stale OS-package pin was added for packages already fixed by the base image. The remaining Python package security overrides still document active transitive dependency or compatibility constraints.
