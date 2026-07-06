# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-draft-model-training:16`

Base image from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}`, resolved to `biweekly.202606.3`.

Base SBOM note: MCR had no attached SBOM referrer, so the exact base image tag was imported to `vulnscan1779267129n11.azurecr.io/sbom-base/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3` for SBOM generation and reviewer verification.

| Vulnerability | Package | Source classification | Base already covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5013832 / GHSA-537c-gmf6-5ccf | `cryptography` | Inherited vulnerable version `46.0.7` from the ACPT base image in `/opt/conda`; this image upgrades the Python 3.13 base conda environment. | No | `context/Dockerfile` | `cryptography>=48.0.1` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base image SBOM shows patched `2.14.2`, but this image's pip dependency layers can override the base environment back to vulnerable `2.12.0`. | Yes for base; no after this image's dependency layers | `context/Dockerfile` | `pydantic-settings>=2.14.2` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Base image SBOM shows patched `1.2.1`, but this image's AzureML/MLflow dependency layers can override the base environment back to vulnerable `1.1.1`. | Yes for base; no after this image's dependency layers | `context/Dockerfile` | `msgpack>=1.2.1` |

Existing pins and overrides audited:

| Pin/override | Current disposition |
| --- | --- |
| `sglang>=0.5.10`, `xgrammar==0.1.32`, `transformers==5.5.4`, `starlette>=1.0.1`, `aiohttp>=3.14.0`, `wandb>=0.27.2`, `pip>=26.1`, `PyJWT>=2.13.0`, `py-rattler>=0.24.0`, `python-dotenv>=1.2.2`, `urllib3>=2.7.0`, `idna>=3.15`, `click>=8.3.3` | Kept; these are existing remediation overrides for prior scan findings or loose transitive dependency ranges, and they are not contradicted by the current base comparison. |
| `cryptography>=48.0.1`, `pydantic-settings>=2.14.2` | Kept; `cryptography` is still vulnerable in the base image, and this image can downgrade `pydantic-settings` during later pip installs. |
| `msgpack>=1.2.1` | Added; base image is patched, but this image can downgrade `msgpack` through AzureML/MLflow dependency installation. |
