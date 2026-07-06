# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-medimageparse-3d-finetune:3`

Base image from `context\Dockerfile`: `mcr.microsoft.com/azureml/openmpi5.0-cuda12.6-ubuntu24.04:{{latest-image-tag}}`; resolved by the azureml-assets pinning utility to `20260621.v1` on 2026-07-06. MCR had no attached SBOM artifact for this base tag, so `base-sbom.json` was generated from a no-op ACR wrapper image that uses the same resolved base: `vulnscan1779267129n6.azurecr.io/temp/base-openmpi5-cuda126-ubuntu2404:20260621-v1`.

| Finding | Package | Source | Base covered? | Files changed | Patched version |
| --- | --- | --- | --- | --- | --- |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | This image directly installed `torch==2.8.0+cu126` in `context\Dockerfile`. | Not inherited from base; `base-sbom.json` contains no `torch` package. | `context\Dockerfile` | `torch==2.10.0` (`2.10.0+cu126` in `sbom.json`) |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | This image directly installed `torch==2.8.0+cu126` in `context\Dockerfile`. | Not inherited from base; `base-sbom.json` contains no `torch` package. | `context\Dockerfile` | `torch==2.10.0` (`2.10.0+cu126` in `sbom.json`) |

Existing CVE pins in `context\Dockerfile` and `context\requirements.txt` were reviewed against the current requested findings. They are not base-image `torch` remediations, and the final VCM scan did not show stale or redundant pins to remove.

Final VCM evaluation:

- `base-sbom.json`: compliant, 0 non-compliant findings.
- `sbom.json`: compliant, 0 non-compliant findings.
