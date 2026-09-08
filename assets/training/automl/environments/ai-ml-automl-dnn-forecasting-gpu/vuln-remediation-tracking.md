# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-forecasting-gpu:50`

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:{{latest-image-tag}}`

`vcm image sbom download` did not find SBOM artifacts attached to the public MCR base tag resolved as `latest`, so the base image was imported to `vulnscan1779267129n9.azurecr.io/temp/base-openmpi5.0-cuda12.4-ubuntu22.04:latest` and scanned there. `base-sbom.json`, `base-vulnerabilities.json`, `sbom.json`, and `vulnerabilities.json` are kept in this directory for manual verification.

| Vulnerability | Package | Source classification | Base covered? | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873121 / USN-8687-1 | `libp11-kit0` | Inherited from base image | Yes; base SBOM has `0.24.0-6ubuntu0.1` | `context/Dockerfile` keeps a conditional upgrade entry for this package | Already at required `0.24.0-6ubuntu0.1` |
| 6873124 / USN-8691-1 | `libattr1` | Inherited from base image | Yes; base SBOM has `1:2.5.1-1ubuntu0.1` | `context/Dockerfile` keeps a conditional upgrade entry for this package | Already at required `1:2.5.1-1ubuntu0.1` |
| 6873125 / USN-8699-1 | `libssh-4` | Inherited from base image | Yes; base SBOM has `0.9.6-2ubuntu0.22.04.8` | `context/Dockerfile` keeps a conditional upgrade entry for this package | Already at required `0.9.6-2ubuntu0.22.04.8` |
| 6873134 / USN-8692-1 | `diffutils` | Inherited from base image | Yes; base SBOM has `1:3.8-0ubuntu2.1` | `context/Dockerfile` keeps a conditional upgrade entry for this package | Already at required `1:3.8-0ubuntu2.1` |
| 6873135 / USN-8697-1 | `coreutils` | Inherited from base image | Yes; base SBOM has `8.32-4.1ubuntu1.4` | `context/Dockerfile` keeps a conditional upgrade entry for this package | Already at required `8.32-4.1ubuntu1.4` |
| 6873148 / USN-8711-1 | `libgcrypt20` | Inherited from base image but still vulnerable; upgraded in this image | No; base SBOM has `1.9.4-3ubuntu3.2` | `context/Dockerfile` upgrades Ubuntu packages in this image layer | Target SBOM has `1.9.4-3ubuntu3.3` |

Existing pins and overrides were audited against the base and remediated scan artifacts. No Python pin was added for these Ubuntu package findings, and no existing Python override was removed because the remaining comments describe active compatibility or security constraints.
