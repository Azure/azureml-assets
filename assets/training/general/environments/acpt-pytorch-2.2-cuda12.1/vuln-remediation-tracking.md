# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

Base SBOM status: `vcm image sbom download` found no attached SBOM artifact for the resolved base image tag, so classification uses the vulnerable install paths from the VCM finding and the Dockerfile dependency layer. `base-sbom.json` could not be produced because no base-image SBOM artifact exists.

Final image SBOM: `sbom.json` for `vulnscan1779267129n11.azurecr.io/public/azureml/curated/acpt-pytorch-2.2-cuda12.1:test-fix@sha256:6ec1ccc78b9eb8ee77ae014fc6fe18076fd1b42901ab9c9e408065a07704901b`.

VCM evaluation: scan `74d179d6-da5f-471e-a476-ffbfeef7d104` completed successfully with `0` non-compliant findings.

| Vulnerability | Package | Source classification | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-qfhq-4f3w-5fph | torch | Inherited from ACPT base image in `/opt/conda/envs/ptca`; no newer CUDA 12.6 ACPT torch base line was available, so this image overrides it. | No | `context/Dockerfile` | `torch==2.10.0+cu126` |
| GHSA-vgrw-7cvw-pwgx | torch | Inherited from ACPT base image in `/opt/conda/envs/ptca`; no newer CUDA 12.6 ACPT torch base line was available, so this image overrides it. | No | `context/Dockerfile` | `torch==2.10.0+cu126` |
| GHSA-6v7p-g79w-8964 | msgpack | Inherited from base conda env (`/opt/conda`) and not directly declared by this asset. | No | `context/Dockerfile` | `msgpack>=1.2.1` |
| GHSA-537c-gmf6-5ccf | cryptography | Reintroduced by AzureML SDK install in the ptca env; this image reapplies the patched wheel after `requirements.txt` and removes stale `cryptography-46.0.7` metadata. | No | `context/Dockerfile` | `cryptography>=48.0.1` (SBOM resolved `49.0.0`) |
| GHSA-4xgf-cpjx-pc3j | pydantic-settings | Inherited from base conda env (`/opt/conda`) and not directly declared by this asset. | No | `context/Dockerfile` | `pydantic-settings>=2.14.2` |

Existing pins/overrides audited:

| Pin/override | Current disposition |
| --- | --- |
| `PyJWT>=2.13.0` | Kept; transitive base-image dependency with documented GHSA remediation. |
| `py-rattler>=0.24.0` | Kept; transitive base conda tooling dependency with documented GHSA remediation. |
| `cryptography>=48.0.1` | Kept and moved to a post-`requirements.txt` override so AzureML SDK dependencies cannot leave stale vulnerable metadata in the ptca env. |
| `pyarrow>=23.0.1` | Kept; documented as a post-install override for `azureml-dataset-runtime`. |
| Other loose security floors (`pip`, `python-dotenv`, `urllib3`, `idna`, `click`, `aiohttp`) | Kept; not flagged in this scan but retained because the base SBOM was unavailable and comments still describe security-floor intent. |
