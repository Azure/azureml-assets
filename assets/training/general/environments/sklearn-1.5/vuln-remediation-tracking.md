# sklearn-1.5 vulnerability remediation tracking

Base image from Dockerfile: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}`, resolved during manual review to digest `sha256:30aba99cdf1b0aedd1615880da33da68a9df90369a36b5084c7287b21d3988fe` / tag `20260901.v1`.

Manual verification artifacts retained in this directory: `base-sbom.json`, `base-vulnerabilities.json`, `sbom.json`, and `vulnerabilities.json`.

Current image SBOM reviewed: `gatestacr.azurecr.io/public/azureml/curated/sklearn-1.5:test-fix` at digest `sha256:8dcffa79ee159aa6a7dd91fafb69e94859ead95107dfe0aa44cfc31d9692727e`; VCM evaluation reported zero non-compliant findings.

| Vulnerability | Package(s) | Source | Base-image status | File(s) changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873134 / USN-8692-1 | diffutils | Base image | Already covered in base SBOM at `1:3.10-1ubuntu0.1` | None | No local pin |
| 5017848 / GHSA-qwm4-qh6w-59xr | pip | This image conda environment | Base `/opt/miniconda` pip already covered at `26.2.1`; this image previously created `/azureml-envs/sklearn-1.5` with `26.1.2` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `26.2.1` |
| 6873148 / USN-8711-1 / CVE-2024-2236 | libgcrypt20 | Base image | Base SBOM still has `1.10.3-2ubuntu0.1`; image layer upgrades it | `context/Dockerfile` | `1.10.3-2ubuntu0.2` |
| 6873141 / USN-8706-1 | zlib1g | Base image | Already covered in base SBOM at `1:1.3.dfsg-3.1ubuntu2.2` | None | No local pin |
| 6873135 / USN-8697-1 | coreutils | Base image | Already covered in base SBOM at `9.4-3ubuntu6.3` | None | No local pin |
| 6873125 / USN-8699-1 | libssh-4 | Base image | Already covered in base SBOM at `0.10.6-2ubuntu0.5` | None | No local pin |
| 6873124 / USN-8691-1 | libattr1 | Base image | Already covered in base SBOM at `1:2.5.2-1ubuntu0.1` | None | No local pin |
| 6873121 / USN-8687-1 | libp11-kit0 | Base image | Already covered in base SBOM at `0.25.3-4ubuntu2.2` | None | No local pin |
| 6873114 / USN-8684-1 | perl, libperl5.38t64, perl-modules-5.38, perl-base | Base image | Already covered in base SBOM at `5.38.2-3.2ubuntu0.4` | None | No local pin |
| GHSA-5rjg-fvgr-3xxf / GHSA-h35f-9h28-mq5c | setuptools | This image conda environment | Base `/opt/miniconda` setuptools already covered at `84.0.0`; image scan treated pip vendored metadata references to `70.3.0` as installed packages | `context/Dockerfile`, `context/conda_dependencies.yaml` | `>=83.0.0`; remove pip vendored metadata from copied env |
| GHSA-6v7p-g79w-8964 | msgpack | This image conda environment | Base `/opt/miniconda` msgpack already covered at `1.2.2`; image scan treated pip vendored metadata references to `1.1.2` as installed packages | `context/Dockerfile`, `context/conda_dependencies.yaml` | `>=1.2.1`; remove pip vendored metadata from copied env |

Existing pin audit:

| Pin/override | Source assessment | Decision |
| --- | --- | --- |
| `/opt/miniconda` `idna>=3.15` and `cryptography>=50.0.0` upgrade | Base SBOM already ships `idna 3.19` and `cryptography 50.0.0` | Removed stale base override |
| Environment `starlette>=1.0.1`, `idna>=3.15`, `aiohttp>=3.14.0`, `cryptography>=50.0.0`, `distributed>=2026.1.0`, `GitPython>=3.1.55`, `msgpack>=1.2.1`, `pyarrow>=23.0.1`, `setuptools>=83.0.0`, `sqlparse>=0.6.0` | Introduced by this image through AzureML/MLflow/distributed dependency resolution rather than the base image | Kept; `setuptools` and `msgpack` are constrained during the main pip install and the runtime stage copies only the completed conda env so stale vulnerable build-layer records are not shipped |
