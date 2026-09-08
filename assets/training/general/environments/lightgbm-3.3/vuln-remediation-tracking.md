# Vulnerability remediation tracking

Image: `public/azureml/curated/lightgbm-3.3:87`

Base image resolved from `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:{{latest-image-tag}}` to `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`.

Base SBOM note: MCR did not publish an SBOM artifact for the resolved base tag, so the same base digest was imported into `vulnscan1779267129n10.azurecr.io/base-comparison/azureml/openmpi5.0-ubuntu24.04:20260901.v1-lightgbm87` and scanned with VCM. The generated comparison files are `base-sbom.json` and `base-vulnerabilities.json`.

Final image SBOM and scan results are retained as `sbom.json` and `vulnerabilities.json`. The final VCM evaluation for `gatestacr.azurecr.io/public/azureml/curated/lightgbm-3.3:test-fix` was compliant.

| ID | Advisory / CVE | Source | Base status | Changed files | Patched / pinned version |
| --- | --- | --- | --- | --- | --- |
| 6873114 | USN-8684-1 Perl vulnerabilities | Base image | Already covered in current base (`perl`, `libperl5.38t64`, `perl-modules-5.38`, and `perl-base` are `5.38.2-3.2ubuntu0.4`) | None | `5.38.2-3.2ubuntu0.4` |
| 6873121 | USN-8687-1 p11-kit vulnerabilities | Base image | Already covered in current base (`libp11-kit0` is `0.25.3-4ubuntu2.2`) | None | `0.25.3-4ubuntu2.2` |
| 6873124 | USN-8691-1 attr vulnerability | Base image | Already covered in current base (`libattr1` is `1:2.5.2-1ubuntu0.1`) | None | `1:2.5.2-1ubuntu0.1` |
| 6873125 | USN-8699-1 libssh vulnerabilities | Base image | Already covered in current base (`libssh-4` is `0.10.6-2ubuntu0.5`) | None | `0.10.6-2ubuntu0.5` |
| 6873134 | USN-8692-1 GNU diffutils vulnerability | Base image | Already covered in current base (`diffutils` is `1:3.10-1ubuntu0.1`) | None | `1:3.10-1ubuntu0.1` |
| 6873135 | USN-8697-1 GNU Core Utilities vulnerabilities | Base image | Already covered in current base (`coreutils` is `9.4-3ubuntu6.3`) | None | `9.4-3ubuntu6.3` |
| 6873141 | USN-8706-1 zlib vulnerability | Base image | Already covered in current base (`zlib1g` is `1:1.3.dfsg-3.1ubuntu2.2`) | None | `1:1.3.dfsg-3.1ubuntu2.2` |
| 6873148 | USN-8711-1 / CVE-2024-2236 Libgcrypt vulnerability | Base image | Current base still has `libgcrypt20` `1.10.3-2ubuntu0.1`; fixed in this image layer | `context/Dockerfile` | `1.10.3-2ubuntu0.2` |
| 5017848 | GHSA-qwm4-qh6w-59xr pip vulnerability | This image conda environment | Base miniconda pip is already above the required version; this image pinned `pip` to a vulnerable `26.1.2` in `/azureml-envs/lightgbm-3.3` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `26.2.1` |
| 5004129 | GHSA-5rjg-fvgr-3xxf / CVE-2025-47273 setuptools vulnerability | This image conda environment | Base has `setuptools` `84.0.0`; this image's conda environment reintroduced `setuptools` `70.3.0` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `84.0.0` |
| 5016030 | GHSA-h35f-9h28-mq5c / CVE-2026-59890 setuptools vulnerability | This image conda environment | Base has `setuptools` `84.0.0`; this image's conda environment reintroduced `setuptools` `70.3.0` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `84.0.0` |
| 5014304 | GHSA-6v7p-g79w-8964 msgpack vulnerability | This image conda environment | Base has `msgpack` `1.2.2`; this image's conda environment reintroduced `msgpack` `1.1.2` | `context/Dockerfile`, `context/conda_dependencies.yaml` | `1.2.2` |

Existing pins and overrides were reviewed against the provided current scan findings and the base SBOM. The Python security overrides already present in the Dockerfile remain image-layer dependency constraints rather than base-image packages, and their comments still describe why the pins exist. The final Dockerfile also removes pip's vendored inventory metadata files because VCM was treating those non-runtime metadata entries as installed vulnerable `setuptools` and `msgpack` packages.
