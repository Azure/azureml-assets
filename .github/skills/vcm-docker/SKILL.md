---
name: vcm-docker
description: "Use when: scanning a locally built Docker image for vulnerabilities using Trivy SBOM output and VCM vulnerability listing."
---

# VCM Docker Vulnerability Scan

## Goal
Use an already built local Docker image, generate an SPDX SBOM with Trivy, and run VCM vulnerability listing on that SBOM.

## Inputs
- Image reference (required): `name:tag`
- Working directory for artifacts (required)
- Optional SBOM output name (default: `sbom.json`)

## Prerequisites
1. Docker daemon is running.
2. Built image exists locally (`docker image inspect <image>` passes).
3. Python and pip are available to install VCM when missing.

## Workflow
1. Ensure `vcm` CLI is available.

```bash
if ! command -v vcm >/dev/null 2>&1; then
   pip install -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/ vienna-container-management
fi
vcm version
```

2. Validate image exists locally.
3. Generate SBOM from the image using Trivy container:

```bash
docker run --rm \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v "$PWD:/work" -w /work \
  ghcr.io/aquasecurity/trivy:latest \
  image \
  --no-progress \
  --format spdx-json \
  --output sbom.json \
  "$IMAGE"
```

4. Confirm `sbom.json` exists in the working directory.
5. Run VCM vulnerability listing:

```bash
vcm image vulnerabilities show --sbom "<absolute-path-to-sbom.json>"
```

6. Parse and summarize vulnerabilities by severity and package.

## Disk/Space Failure Handling
If Trivy/VCM fails with disk-space errors:
1. Stop running containers: `docker stop $(docker ps -q)` (or platform equivalent).
2. Run Docker cleanup (least to most aggressive):
   - `docker image prune -f`
   - `docker builder prune -f`
   - `docker container prune -f`
   - `docker volume prune -f`
   - `docker system prune -f`
3. Retry SBOM + VCM scan once.
4. If still failing on Linux and user approves, use the Docker data-dir relocation workaround:
   - Stop Docker service.
   - Move `/var/lib/docker` to `/mnt/docker`.
   - Symlink `/var/lib/docker` -> `/mnt/docker`.
   - Start Docker service.
   - Verify with `docker ps` and `docker images`.
   - Retry scan.

## Platform Notes
- The `/var/run/docker.sock` mount and `/var/lib/docker` relocation are Linux-specific.
- On Windows, use equivalent Docker Desktop cleanup and storage settings instead of Linux service commands.

## Output
Return:
1. Image scanned
2. SBOM path
3. VCM findings summary (critical/high/medium/low)
4. Raw command status for Trivy and VCM
5. Cleanup/remediation actions attempted
6. If unresolved, exact blocker and next manual command(s)
