# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-hf-nlp-gpu:125`
Base image checked from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

Base SBOM note: `vcm image sbom download` returned no SBOM artifacts for the base image tag, so base package comparison could not be completed from a base SBOM.

| Finding | Package(s) | Source classification | Base already covered? | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-qfhq-4f3w-5fph / 5014620 | `torch` | Inherited from ACPT base image in `ptca`; overridden by this image for CUDA 12.6 compatibility | Unknown; base SBOM unavailable | `context/Dockerfile` | `torch==2.10.0+cu126` with matching `torchvision==0.25.0+cu126`, `torchaudio==2.10.0+cu126` |
| GHSA-vgrw-7cvw-pwgx / 5014615 | `torch` | Inherited from ACPT base image in `ptca`; overridden by this image for CUDA 12.6 compatibility | Unknown; base SBOM unavailable | `context/Dockerfile` | `torch==2.10.0+cu126` with matching `torchvision==0.25.0+cu126`, `torchaudio==2.10.0+cu126` |
| GHSA-6v7p-g79w-8964 / 5014304 | `msgpack` | Inherited from ACPT base image in base conda env; overridden by this image | Unknown; base SBOM unavailable | `context/Dockerfile` | `msgpack>=1.2.1` |
| USN-8495-1 / 6035947 | `libnghttp2-14` | Inherited Ubuntu package from base image; upgraded by this image | Unknown; base SBOM unavailable | `context/Dockerfile` | `1.43.0-1ubuntu0.4` or newer from Ubuntu repositories |
| USN-8487-1 / 6035933 | `curl`, `libcurl3-gnutls`, `libcurl4` | Inherited Ubuntu packages from base image; upgraded by this image | Unknown; base SBOM unavailable | `context/Dockerfile` | `7.81.0-1ubuntu1.25` or newer from Ubuntu repositories |
| USN-8480-1 / 6035919 | `libsqlite3-0` | Inherited Ubuntu package from base image; upgraded by this image | Unknown; base SBOM unavailable | `context/Dockerfile` | `3.37.2-2ubuntu0.6` or newer from Ubuntu repositories |

Existing pins and overrides were reviewed against the provided vulnerability list. The existing torch override is still needed for the listed torch findings. No existing pin was removed because the base SBOM was unavailable and the provided current scan data did not show that any existing override is stale.
