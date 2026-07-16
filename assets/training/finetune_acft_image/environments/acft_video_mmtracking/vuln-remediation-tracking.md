# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

Comparison artifacts retained for review:
- `base-sbom.json`
- `base-vulnerabilities.json`
- `sbom.json`

| Finding | Package(s) | Source classification | Base image status | Changed file(s) | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5014615 / GHSA-vgrw-7cvw-pwgx / CVE-2025-2999 | torch | Inherited from base image but overridden in this image | Base has `torch 2.8.0+cu126`, vulnerable | `context/Dockerfile` | `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0` |
| 5014620 / GHSA-qfhq-4f3w-5fph / CVE-2025-3001 | torch | Inherited from base image but overridden in this image | Base has `torch 2.8.0+cu126`, vulnerable | `context/Dockerfile` | `torch==2.10.0`, `torchvision==0.25.0`, `torchaudio==2.10.0` |
| 5015183 / GHSA-hwpq-hmq9-wj77 | onnx | Transitive dependency of `azureml-acft-accelerator` in this image | Base already has `onnx 1.22.0`, covered, but image requirements can downgrade it | `context/Dockerfile` | `onnx==1.22.0` |
| 5015223 / GHSA-wf93-45jw-7689 | pip | Inherited from base image | Base already has `pip 26.1.2` in conda root and ptca envs, covered | None | Base `pip 26.1.2` |
| 5014304 / GHSA-6v7p-g79w-8964 | msgpack | Inherited from base image | Base already has `msgpack 1.2.1`, covered | None | Base `msgpack 1.2.1` |
| 5013832 / GHSA-537c-gmf6-5ccf | cryptography | Transitive dependency of `azureml-core`/`pyOpenSSL` in this image | Base already has `cryptography 48.0.1`, covered, but image requirements can downgrade it | `context/Dockerfile` | `cryptography>=48.0.1,<49` |
| 6035961 / USN-8509-1 | libpython3.10-stdlib, python3.10, python3.10-minimal, libpython3.10-minimal | Inherited from base image | Base already has `3.10.12-1~22.04.16`, covered | None | Base `3.10.12-1~22.04.16` |
| 6035957 / USN-8510-1 | tar | Inherited from base image | Base already has `1.34+dfsg-1ubuntu0.1.22.04.4`, covered | None | Base `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035952 / USN-8512-1 | gzip | Inherited from base image | Base already has `1.10-4ubuntu4.2`, covered | None | Base `1.10-4ubuntu4.2` |
| 6035947 / USN-8495-1 | libnghttp2-14 | Inherited from base image | Base already has `1.43.0-1ubuntu0.4`, covered | None | Base `1.43.0-1ubuntu0.4` |
| 6035933 / USN-8487-1 | curl, libcurl3-gnutls, libcurl4 | Inherited from base image | Base already has `7.81.0-1ubuntu1.25`, covered | None | Base `7.81.0-1ubuntu1.25` |
| 6035919 / USN-8480-1 | libsqlite3-0 | Inherited from base image | Base already has `3.37.2-2ubuntu0.6`, covered | None | Base `3.37.2-2ubuntu0.6` |
