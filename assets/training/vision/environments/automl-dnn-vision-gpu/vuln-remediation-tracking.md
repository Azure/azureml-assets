# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu118-py310-torch271:biweekly.202601.1`

VCM did not find an attached SBOM artifact for the base image in MCR, and VCM SBOM generation cannot attach to `mcr.microsoft.com` from this environment. `base-sbom.json` and `base-vulnerabilities.json` were generated with Trivy as fallback reviewer evidence.

Current verification image: `gatestacr.azurecr.io/public/azureml/curated/automl-dnn-vision-gpu:test-fix@sha256:6f88479185236b640422748ba02b40bf395df5c78bbb6f1f3a120535a0d14de2`

VCM evaluation for the current verification image completed as compliant with 0 non-compliant findings on 2026-09-08. `sbom.json` and `vulnerabilities.json` contain the downloaded SBOM and detailed findings for reviewer verification.

| Finding | Package | Source | Base covered? | Changed files | Patched version |
| --- | --- | --- | --- | --- | --- |
| USN-8691-1 / CVE-2026-54371 | `libattr1` | Base image OS package | No; base has `1:2.5.1-1build1` | `context/Dockerfile` | `1:2.5.1-1ubuntu0.1` |
| USN-8687-1 / CVE-2026-13757, CVE-2026-18938 | `libp11-kit0` | Base image OS package | No; base has `0.24.0-6build1` | `context/Dockerfile` | `0.24.0-6ubuntu0.1` |
| USN-8699-1 / CVE-2026-59843, CVE-2026-59845, CVE-2026-59846, CVE-2026-59847, CVE-2026-59848, CVE-2026-59850 | `libssh-4` | Base image OS package | No; base has `0.9.6-2ubuntu0.22.04.5` | `context/Dockerfile` | `0.9.6-2ubuntu0.22.04.8` |
| USN-8692-1 / CVE-2026-53910 | `diffutils` | Base image OS package | No; base has `1:3.8-0ubuntu2` | `context/Dockerfile` | `1:3.8-0ubuntu2.1` |
| USN-8697-1 / CVE-2025-5278 | `coreutils` | Base image OS package | No; base has `8.32-4.1ubuntu1.2` | `context/Dockerfile` | `8.32-4.1ubuntu1.4` |
| USN-8711-1 / CVE-2024-2236, CVE-2026-41989 | `libgcrypt20` | Base image OS package | No; base has `1.9.4-3ubuntu3` | `context/Dockerfile` | `1.9.4-3ubuntu3.3` |
| USN-8710-1 / CVE-2026-63381, CVE-2026-63382, CVE-2026-63383, CVE-2026-63384, CVE-2026-63385 | `libevent-core-2.1-7` | Base image OS package | No; base has `2.1.12-stable-1build3` | `context/Dockerfile` | `2.1.12-stable-1ubuntu0.1` |
