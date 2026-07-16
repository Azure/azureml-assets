# Vulnerability remediation tracking

Base image: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

The MCR base image did not expose a downloadable VCM SBOM artifact during this run. The available generated base SBOM for the same digest (`sha256:2c139ba03468a302ef9c11507dfd064ecd7ee4bd993cb64670bbe72db69b10ff`) is kept as `base-sbom.json`, with detailed findings in `base-vulnerabilities.json`.

Validation image: `vulnscan1779267129n5.azurecr.io/public/azureml/curated/acft-medimageinsight-embedding-generator:test-fix` (`sha256:3088d23ba54bde5ea5a51bf99c302c22c8844b78c32adfd69fb4cb5cb93be1a6`). The VCM evaluation of `sbom.json` completed compliant with zero non-compliant findings.

| Finding | Package(s) | Source | Base covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 6035952 / USN-8512-1 | gzip | Base image Ubuntu package | Yes | None | Base has `1.10-4ubuntu4.2` |
| 6035957 / USN-8510-1 | tar | Base image Ubuntu package | Yes | None | Base has `1.34+dfsg-1ubuntu0.1.22.04.4` |
| 6035961 / USN-8509-1 | libpython3.10-stdlib, python3.10, python3.10-minimal, libpython3.10-minimal | Base image Ubuntu packages | Yes | None | Base has `3.10.12-1~22.04.16` |
| 5015223 / GHSA-wf93-45jw-7689 | pip | Base image Python packages | Yes | None | Base has `26.1.2` in ptca and base conda Python |
| 5014620 / GHSA-qfhq-4f3w-5fph | torch | Base image ptca Python package | No | `context/Dockerfile` | `torch==2.10.0+cu126` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | torch | Base image ptca Python package | No | `context/Dockerfile` | `torch==2.10.0+cu126` |
| 5014304 / GHSA-6v7p-g79w-8964 | msgpack | Base image Python package; image requirements can downgrade via AzureML/MLflow dependency resolution | Base has patched version, image keeps override | `context/Dockerfile` | `msgpack>=1.2.1` |
| 6035947 / USN-8495-1 | libnghttp2-14 | Base image Ubuntu package | Yes | None | Base has `1.43.0-1ubuntu0.4` |
| 6035933 / USN-8487-1 | curl, libcurl3-gnutls, libcurl4 | Base image Ubuntu packages | Yes | None | Base has `7.81.0-1ubuntu1.25` |
| 6035919 / USN-8480-1 | libsqlite3-0 | Base image Ubuntu package | Yes | None | Base has `3.37.2-2ubuntu0.6` |
