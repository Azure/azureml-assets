# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`

The base image SBOM was requested from MCR with `vcm image sbom download`, but MCR returned no SBOM referrer artifact for the resolved base tag. `base-sbom.json` records that failed lookup. The rebuilt image SBOM is saved as `sbom.json`.

Validation image: `vulnscan1779267129n5.azurecr.io/public/azureml/curated/acft-medimageparse-finetune:16-test-fix`

VCM evaluation result: compliant, with 0 non-compliant findings.

| Vulnerability | Package | Source | Base image already covered? | File(s) changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | Base Python 3.13 conda environment | No; vulnerable `2.12.0` was reported under `/opt/conda/lib/python3.13` | `context/Dockerfile` | `>=2.14.2` |
| GHSA-6v7p-g79w-8964 | `msgpack` | Base Python 3.13 conda environment / transitive Python dependency | No; vulnerable `1.1.1` was reported under `/opt/conda/lib/python3.13` | `context/Dockerfile` | `>=1.2.1` |
| GHSA-vgrw-7cvw-pwgx | `torch` | Inherited from the `torch280` ptca base environment | No; vulnerable `2.8.0+cu126` was reported under `/opt/conda/envs/ptca` | `context/Dockerfile` | `2.10.0+cu126` |
| GHSA-qfhq-4f3w-5fph | `torch` | Inherited from the `torch280` ptca base environment | No; vulnerable `2.8.0+cu126` was reported under `/opt/conda/envs/ptca` | `context/Dockerfile` | `2.10.0+cu126` |

Final SBOM versions: `pydantic-settings` 2.14.2, `msgpack` 1.2.1, `torch` 2.10.0+cu126, `torchvision` 0.25.0+cu126, and `torchaudio` 2.10.0+cu126.
