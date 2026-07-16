# Vulnerability remediation tracking

Image: `public/azureml/curated/acpt-pytorch-2.2-cuda12.1:54`

Base image resolved from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202607.1`

Note: `vcm image sbom download` found no SBOM artifact for the resolved base image, and `vcm image sbom generate` cannot attach SBOMs to `mcr.microsoft.com` images from this ACR workflow. Base classification used direct ACR build inspection of package versions from the resolved base digest `sha256:2c139ba03468a302ef9c11507dfd064ecd7ee4bd993cb64670bbe72db69b10ff`.

Final verification image digest: `sha256:0d25f2db24df4bd1ba5d8df07e26e4a3ed808b3b260529ba1ef6ac6aee942024`; final VCM evaluation was compliant and `vulnerabilities.json` did not contain the requested finding IDs.

| Finding | Package | Source | Base covered? | Remediation | Files changed |
| --- | --- | --- | --- | --- | --- |
| 6035912 / USN-8477-1 | `tar` | Base image OS package | Yes: base has `1.34+dfsg-1ubuntu0.1.22.04.4` | No pin added; base is above required `1.34+dfsg-1ubuntu0.1.22.04.3` | None |
| 6035919 / USN-8480-1 | `libsqlite3-0` | Base image OS package | Yes: base has `3.37.2-2ubuntu0.6` | No pin added; base is at required version | None |
| 6035933 / USN-8487-1 | `curl`, `libcurl3-gnutls`, `libcurl4` | Base image OS packages | Yes: base has `7.81.0-1ubuntu1.25` | No pin added; base is at required version | None |
| 6035947 / USN-8495-1 | `libnghttp2-14` | Base image OS package | Yes: base has `1.43.0-1ubuntu0.4` | No pin added; base is at required version | None |
| 5013832 / GHSA-537c-gmf6-5ccf | `cryptography` | `ptca` env dependency re-resolved by this image's `requirements.txt` install | Partially: base root env has patched `cryptography`; `ptca` env was downgraded to `46.0.7` after requirements installation | Added final `cryptography>=48.0.1` post-requirements pin | `context/Dockerfile` |
| 5014292 / GHSA-4xgf-cpjx-pc3j | `pydantic-settings` | This image can re-resolve root env via conda/pip operations | Yes in base root env: `2.14.2`; pin retained to prevent downgrade | Added/retained `pydantic-settings>=2.14.2` in root env pip upgrade | `context/Dockerfile` |
| 5014304 / GHSA-6v7p-g79w-8964 | `msgpack` | This image can re-resolve root env via conda/pip operations | Yes in base root env: `1.2.1`; pin retained to prevent downgrade | Added/retained `msgpack>=1.2.1` in root env pip upgrade | `context/Dockerfile` |
| 5014615 / GHSA-vgrw-7cvw-pwgx | `torch` | Inherited vulnerable package from ACPT `ptca` base env | No: base `ptca` env has `2.8.0+cu126` | Upgraded to `torch==2.10.0+cu126` | `context/Dockerfile` |
| 5014620 / GHSA-qfhq-4f3w-5fph | `torch` | Inherited vulnerable package from ACPT `ptca` base env | No: base `ptca` env has `2.8.0+cu126` | Upgraded to `torch==2.10.0+cu126`; upgraded `torchvision==0.25.0+cu126` and `torchaudio==2.10.0+cu126` for stack compatibility | `context/Dockerfile` |
