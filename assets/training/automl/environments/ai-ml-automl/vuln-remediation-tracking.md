# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-gpu:51`

Base image checked: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260621.v1`

Base SBOM note: MCR did not publish an SBOM referrer for this tag, so the same base image digest was imported to `vulnscan1779267129n4.azurecr.io/temp/base-openmpi5.0-ubuntu24.04:20260621.v1` and scanned there. The generated base SBOM is saved as `base-sbom.json`; the remediated image SBOM is saved as `sbom.json`.

| Advisory | Package | Source classification | Base already covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-qfhq-4f3w-5fph | `torch` | Introduced by this image via the AutoML conda environment pip install. | Not inherited from the current base image; `base-sbom.json` has no `torch` package. | `context/Dockerfile` | `torch==2.10.0` |
| GHSA-vgrw-7cvw-pwgx | `torch` | Introduced by this image via the AutoML conda environment pip install. | Not inherited from the current base image; `base-sbom.json` has no `torch` package. | `context/Dockerfile` | `torch==2.10.0` |
| GHSA-6v7p-g79w-8964 | `msgpack` | Inherited from the base image Python prefix. | Yes; `base-sbom.json` has `msgpack==1.2.1`. | None | No image-layer pin; base-covered at `1.2.1`. |
| GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Inherited from the base image Python prefix. | Yes; `base-sbom.json` has `pydantic-settings==2.14.2`. | None | No image-layer pin; base-covered at `2.14.2`. |
| GHSA-537c-gmf6-5ccf | `cryptography` | Inherited from the base image Python prefix for the reported `/opt/miniconda` path; this image also keeps its existing AutoML-environment override. | Yes for `/opt/miniconda`; `base-sbom.json` has `cryptography==49.0.0`. | `context/Dockerfile` | No base-prefix pin; base-covered at `49.0.0`. Existing AutoML override remains `cryptography>=48.0.1`. |

Existing base-prefix overrides for `PyJWT`, `idna`, `py-rattler`, and `cryptography` were reviewed against `base-sbom.json`. The base image already ships patched `PyJWT 2.13.0`, `idna 3.18`, `cryptography 49.0.0`, and no vulnerable `py-rattler`, so those base-prefix upgrade pins and their stale comments were removed rather than carried forward.

Existing AutoML-environment overrides for `distributed`, `bokeh`, `cryptography`, `onnx`, `pillow`, `python-dotenv`, and `pyarrow` remain because they apply to packages introduced or resolved by this image's AutoML dependency installation, not to already-covered base-image packages.

Validation: `vcm image vulnerabilities evaluate --sbom sbom.json` completed successfully and reported the image compliant with zero non-compliant findings.
