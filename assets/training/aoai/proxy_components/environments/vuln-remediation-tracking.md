# Vulnerability remediation tracking

Image: `public/azureml/curated/aoai-data-upload-finetune:52`

Base image: `mcr.microsoft.com/azureml/openmpi5.0-ubuntu24.04:20260901.v1`

Generated verification artifacts kept beside this file. The base image did not publish an attached SBOM artifact for direct download, so `base-sbom.json` was generated for manual comparison.
- `base-sbom.json`
- `base-vulnerabilities.json`
- `sbom.json`
- `vulnerabilities.json`

| CVE/advisory | Package(s) | Source | Base status | This-image action | Changed files |
| --- | --- | --- | --- | --- | --- |
| 6873121 / USN-8687-1 | `libp11-kit0` | Base image | Already covered in base at `0.25.3-4ubuntu2.2` | No pin added | None |
| 6873114 / USN-8684-1 | `perl`, `libperl5.38t64`, `perl-modules-5.38`, `perl-base` | Base image | Already covered in base at `5.38.2-3.2ubuntu0.4` | No pin added | None |
| 6873124 / USN-8691-1 | `libattr1` | Base image | Already covered in base at `1:2.5.2-1ubuntu0.1` | No pin added | None |
| 6873125 / USN-8699-1 | `libssh-4` | Base image | Already covered in base at `0.10.6-2ubuntu0.5` | No pin added | None |
| 6873134 / USN-8692-1 | `diffutils` | Base image | Already covered in base at `1:3.10-1ubuntu0.1` | No pin added | None |
| 6873135 / USN-8697-1 | `coreutils` | Base image | Already covered in base at `9.4-3ubuntu6.3` | No pin added | None |
| 6873141 / USN-8706-1 | `zlib1g` | Base image | Already covered in base at `1:1.3.dfsg-3.1ubuntu2.2` | No pin added | None |
| 6873148 / USN-8711-1 / CVE-2024-2236 | `libgcrypt20` | Base image | Base still has `1.10.3-2ubuntu0.1` and is flagged by VCM | Pin inherited package to `1.10.3-2ubuntu0.2` during image build | `context/Dockerfile` |

Pin audit:

| Pin or override | Source | Decision |
| --- | --- | --- |
| `GitPython>=3.1.58` | Introduced transitively by `azureml-mlflow` | Kept; not present in base and remains an image-layer transitive security floor. |
| `sqlparse>=0.6.0` | Introduced transitively by `azureml-mlflow` | Kept; not present in base and remains an image-layer transitive security floor. |
| `h2>=4.4.1` | Base miniconda HTTP stack | Removed; current base already ships `h2` `4.4.1`, and the final image keeps that version without an image-layer override. |
| `cryptography>=50.0.0 --no-deps` | `azureml-mlflow`/`msal` dependency chain | Kept; `azureml-mlflow` constrains `cryptography` below the patched floor during requirements install, so this image-layer override is still needed. |

Final VCM evaluation for `gatestacr.azurecr.io/public/azureml/curated/aoai-data-upload-finetune:test-fix`: compliant, with zero non-compliant findings on 2026-09-08.
