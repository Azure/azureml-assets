# Vulnerability remediation tracking

| Vulnerability | Package | Source | Base image covered? | Remediation | Changed files |
| --- | --- | --- | --- | --- | --- |
| GHSA-xrqw-3rrv-vx5w / 5017847 | transformers | This image installs the direct Python dependency through `context/requirements.txt` and the final Dockerfile upgrade layer. | Not inherited from base image in `base-sbom.json`. | Raised the allowed version to `>=5.10.0`; latest scan artifact resolves `transformers` to 5.16.1. | `context/requirements.txt`, `context/Dockerfile` |
| CVE-2024-2236 / USN-8711-1 / 6873148 | libgcrypt20 | Inherited from `mcr.microsoft.com/azureml/openmpi5.0-cuda12.8-ubuntu24.04`. | No. `base-vulnerabilities.json` reports `libgcrypt20` 1.10.3-2ubuntu0.1 below required 1.10.3-2ubuntu0.2. | Explicitly install/upgrade `libgcrypt20`; latest scan artifact resolves it to 1.10.3-2ubuntu0.2. | `context/Dockerfile` |
