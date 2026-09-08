# Vulnerability remediation tracking

| Vulnerability | Package | Source | Base image covered? | Changed files | Patched/pinned version |
| --- | --- | --- | --- | --- | --- |
| USN-8711-1 / 6873148 / CVE-2024-2236 | libgcrypt20 | Inherited from `mcr.microsoft.com/azureml/openmpi5.0-cuda12.4-ubuntu22.04:20260901.v1` and upgraded in this image's OS package remediation layer | No. `base-sbom.json` shows `1.9.4-3ubuntu3.2` and `base-vulnerabilities.json` reports 6873148. `sbom.json` shows this image has `1.9.4-3ubuntu3.3`, and `vulnerabilities.json` no longer reports 6873148. | `context/Dockerfile` | `1.9.4-3ubuntu3.3` via `apt-get install --only-upgrade libgcrypt20=1.9.4-3ubuntu3.3` |
