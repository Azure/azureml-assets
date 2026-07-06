# Vulnerability remediation tracking

Image: `public/azureml/curated/acft-rft-training:20`

Base image from Dockerfile: `mcr.microsoft.com/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:{{latest-image-tag:biweekly\.\d{6}\.\d{1}.*}}`, resolved for verification to `biweekly.202606.3`.

Base SBOM note: `vcm image sbom download` found no SBOM referrer on MCR, so the same base tag was imported to `vulnscan1779267129n4.azurecr.io/sbom-base/aifx/acpt/stable-ubuntu2204-cu126-py310-torch280:biweekly.202606.3`, scanned with Trivy, and attached as an SBOM artifact. `base-sbom.json` is kept beside this file for manual verification.

| VCM ID | Package | Source | Base already covered? | Files changed | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| 5013672 | vllm | This image direct Dockerfile install in ptca env | No; not present in base SBOM | `context/Dockerfile` | `vllm==0.22.0` |
| 5013832 | cryptography | Base image has vulnerable base-env copy; this image also installs ptca copy through AzureML dependencies | No; base SBOM has `cryptography 46.0.7` in `/opt/conda/lib/python3.13` | `context/Dockerfile` | `cryptography>=48.0.1` in base and ptca envs |
| 5013910 | vllm | This image direct Dockerfile install in ptca env | No; not present in base SBOM | `context/Dockerfile` | `vllm==0.22.0` |
| 5013924 | vllm | This image direct Dockerfile install in ptca env | No; not present in base SBOM | `context/Dockerfile` | `vllm==0.22.0` |
| 5014292 | pydantic-settings | Base image already has patched base-env copy; this image introduces vulnerable ptca copy via transitive dependencies | Base copy covered: `pydantic-settings 2.14.2`; ptca copy still remediated here | `context/Dockerfile` | `pydantic-settings>=2.14.2` in ptca env |
| 5014304 | msgpack | Base image | Yes; base SBOM has `msgpack 1.2.1` in `/opt/conda/lib/python3.13` | None | No image-layer pin added |
| 5014600 | jackson-databind | This image installs Ray, whose shaded jar contains `jackson-databind 2.18.6` | No; Ray jar not present in base SBOM | `context/Dockerfile`, `context/requirements.txt`, `context/patch_ray_jackson_databind.py` | Ray shaded jar patched to `jackson-databind 2.19.4` |
| 5014605 | jackson-databind | This image installs Ray, whose shaded jar contains `jackson-databind 2.18.6` | No; Ray jar not present in base SBOM | `context/Dockerfile`, `context/requirements.txt`, `context/patch_ray_jackson_databind.py` | Ray shaded jar patched to `jackson-databind 2.19.4` |
| 5014607 | jackson-databind | This image installs Ray, whose shaded jar contains `jackson-databind 2.18.6` | No; Ray jar not present in base SBOM | `context/Dockerfile`, `context/requirements.txt`, `context/patch_ray_jackson_databind.py` | Ray shaded jar patched to `jackson-databind 2.19.4` |
| 5014608 | jackson-databind | This image installs Ray, whose shaded jar contains `jackson-databind 2.18.6` | No; Ray jar not present in base SBOM | `context/Dockerfile`, `context/requirements.txt`, `context/patch_ray_jackson_databind.py` | Ray shaded jar patched to `jackson-databind 2.19.4` |
