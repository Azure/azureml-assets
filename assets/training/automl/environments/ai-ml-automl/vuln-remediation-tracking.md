# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-text-gpu:56`

Base image from target Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}`, resolved in the validation build to `biweekly.202608.2`.

Base SBOM/finding files used for manual verification:

- `ai-ml-automl-dnn-text-gpu/base-sbom.json`
- `ai-ml-automl-dnn-text-gpu/base-vulnerabilities.json`

Remediated image finding files used for manual verification:

- `ai-ml-automl-dnn-text-gpu/sbom.json` when an SBOM referrer is available
- `ai-ml-automl-dnn-text-gpu/vulnerabilities.json`

Note: `vcm image sbom download` found no SBOM referrer for the latest matching base tag in MCR and no SBOM referrer for the rebuilt ACR tag during this run. The asset-local base evidence for `biweekly.202608.2` was used for classification, and direct VCM vulnerability evaluation was used for the rebuilt ACR tag.

| Finding | Package(s) | Source classification | Base image status | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873121 / USN-8687-1 | `libp11-kit0` | Base image | Base SBOM has vulnerable `0.24.0-6build1`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `0.24.0-6ubuntu0.1` |
| 6873124 / USN-8691-1 | `libattr1` | Base image | Base SBOM has vulnerable `1:2.5.1-1build1`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `1:2.5.1-1ubuntu0.1` |
| 6873125 / USN-8699-1 | `libssh-4` | Base image | Base SBOM has vulnerable `0.9.6-2ubuntu0.22.04.7`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `0.9.6-2ubuntu0.22.04.8` |
| 6873134 / USN-8692-1 | `diffutils` | Base image | Base SBOM has vulnerable `1:3.8-0ubuntu2`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `1:3.8-0ubuntu2.1` |
| 6873135 / USN-8697-1 | `coreutils` | Base image | Base SBOM has vulnerable `8.32-4.1ubuntu1.3`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `8.32-4.1ubuntu1.4` |
| 6873148 / USN-8711-1 | `libgcrypt20` | Base image | Base SBOM has vulnerable `1.9.4-3ubuntu3.2`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `1.9.4-3ubuntu3.3` |
| 6873149 / USN-8710-1 | `libevent-core-2.1-7` | Base image | Base SBOM has vulnerable `2.1.12-stable-1build3`; this image's apt-upgrade layer remediates it. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `2.1.12-stable-1ubuntu0.1` |
| 5017847 / GHSA-xrqw-3rrv-vx5w | `transformers` | This image; installed through `azureml-automl-dnn-nlp` | Not applicable; package is introduced in this image's AzureML env. | `ai-ml-automl-dnn-text-gpu\context\Dockerfile` | `transformers[sentencepiece,torch]==5.10.0` |

Existing pin audit: the stale `pyarrow==14.0.2` override was removed because the current AzureML runtime resolves `pyarrow==17.0.0`, the current scan did not flag `pyarrow`, and the downgrade caused a dependency conflict with `azureml-train-automl-runtime`.
