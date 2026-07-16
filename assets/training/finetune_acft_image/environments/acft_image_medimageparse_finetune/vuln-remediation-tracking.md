# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

| Finding | Package(s) | Source | Base covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035952 / USN-8512-1 | `gzip` | Ubuntu package inherited from the base image and upgraded in this image layer | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `1.10-4ubuntu4.2` |
| 6035957 / USN-8510-1 | `tar` | Ubuntu package inherited from the base image and upgraded in this image layer | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035961 / USN-8509-1 | `libpython3.10-stdlib`, `python3.10`, `python3.10-minimal`, `libpython3.10-minimal` | Ubuntu packages inherited from the base image and upgraded in this image layer | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `3.10.12-1~22.04.16` |
| 5015223 / GHSA-wf93-45jw-7689 | `pip` | Preinstalled Python packaging tool in the base and `ptca` conda environments | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `26.1.2` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | PyTorch runtime inherited from the torch 2.8 base image | Base vulnerabilities reported `torch 2.8.0+cu126`; final image scan did not report it | `context/Dockerfile` | `2.10.0+cu126` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | PyTorch runtime inherited from the torch 2.8 base image | Base vulnerabilities reported `torch 2.8.0+cu126`; final image scan did not report it | `context/Dockerfile` | `2.10.0+cu126` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | Transitive dependency in the base Python stack and MLflow dependencies | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `>=1.2.1` |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Ubuntu package inherited from the base image and upgraded in this image layer | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `1.43.0-1ubuntu0.4` |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Ubuntu packages inherited from the base image and upgraded in this image layer | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `7.81.0-1ubuntu1.25` |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Ubuntu package inherited from the base image and upgraded in this image layer | Base vulnerabilities did not report this finding; final image scan did not report it | `context/Dockerfile` | `3.37.2-2ubuntu0.6` |

Notes:
- `base-sbom.json` and `base-vulnerabilities.json` are retained untracked for manual verification.
- The final image SBOM and findings are retained as untracked `sbom.json` and `vulnerabilities.json`. VCM evaluation reported the remediated image compliant with zero non-compliant findings.
- Existing pins in `context/Dockerfile` and `context/requirements.txt` were reviewed; active security overrides were retained and stale pip 26.1.1 pins were updated to 26.1.2.
