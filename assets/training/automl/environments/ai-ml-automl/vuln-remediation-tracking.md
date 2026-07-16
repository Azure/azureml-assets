# Vulnerability remediation tracking

Image: `public/azureml/curated/ai-ml-automl-dnn-vision-gpu:51`

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3` (resolved from `{{latest-image-tag}}`). The base image SBOM and findings are retained under `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\base-sbom.json` and `base-vulnerabilities.json` for review.

Build context audit: `context\Dockerfile` is the Dockerfile under the dnn-vision GPU asset. No conda YAML or requirements.txt files are present in that context directory.

Post-build scan: VCM evaluation for `vulnscan1779267129n13.azurecr.io/public/azureml/curated/ai-ml-automl-dnn-vision-gpu:test-fix` completed compliant with 0 non-compliant findings. The fixed image digest is `sha256:325883bc5b8559e331d0a6ca0d113b18f880d89bc9e14ebe1d9eb41672212cf7`. The fixed image SBOM is retained as `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\sbom.json`, and detailed fixed-image findings are retained as `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\vulnerabilities.json`.

| Finding | Package(s) | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Inherited from the ACPT base image | No; base SBOM has `3.37.2-2ubuntu0.5` | `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | upgraded by `apt-get upgrade`/reinstall to at least `3.37.2-2ubuntu0.6` |
| 6035912 / USN-8477-1 | `tar` | Inherited from the ACPT base image | Yes; base SBOM has `1.34+dfsg-1ubuntu0.1.22.04.3` | None | No local override |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Inherited from the ACPT base image and installed into this image's AutoML conda env | No; base SBOM has `2.8.0+cu126` | `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | `torch>=2.10.0` in both the AutoML and `ptca` environments |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Inherited from the ACPT base image and installed into this image's AutoML conda env | No; base SBOM has `2.8.0+cu126` | `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | `torch>=2.10.0` in both the AutoML and `ptca` environments |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Inherited from the ACPT base image | Yes; base SBOM has `1.2.1` | None | No local override |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Inherited from the ACPT base image | Yes; base SBOM has `2.14.2` | None | No local override |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Inherited from the ACPT base image | No; base SBOM has `1.43.0-1ubuntu0.3` | `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | upgraded by `apt-get upgrade`/reinstall to at least `1.43.0-1ubuntu0.4` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Inherited from the ACPT base image | No; base SBOM has `7.81.0-1ubuntu1.24` | `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\context\Dockerfile` | upgraded by `apt-get upgrade`/reinstall to at least `7.81.0-1ubuntu1.25` |

Existing pins and overrides were audited against the base SBOM. No new pins were added for `msgpack`, `pydantic-settings`, or `tar` because the resolved base image already contains the required patched versions for the requested findings. Existing security overrides for unrelated findings remain in place only where the current base SBOM still contains vulnerable packages or the AzureML parent dependency constraints do not yet carry the patched floor.

Build stability: `azureml-inference-server-http==1.4.1` is pinned in `assets\training\automl\environments\ai-ml-automl-dnn-vision-gpu\context\Dockerfile` because it is the supported `azureml-defaults~=1.62.0` dependency line and avoids resolving newer inference-server dependencies unrelated to these CVEs.
