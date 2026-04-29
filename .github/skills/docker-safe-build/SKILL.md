---
name: docker-safe-build
description: "Use when: building Docker images locally for vulnerability workflows, especially when disk pressure or no-space errors can occur. Includes safe cleanup and failure reporting."
---

# Docker Safe Build

## Goal
Build Docker images locally for subsequent security scanning while avoiding host disk exhaustion and leaving the system in a recoverable state.

## Inputs
- Docker build context path
- Dockerfile path (optional)
- Image tag (required)
- Build args (optional)

## Pre-checks
1. Verify Docker CLI is available.
2. Verify Docker daemon is reachable.
3. Check available disk space on the Docker storage volume.
4. If free space is critically low before build, perform safe cleanup before retrying build.

## Build Workflow
1. Build image with explicit tag.
2. On success, report:
   - Built image tag
   - Image ID
   - Image size
3. On failure due to disk-space issues:
   - Detect common patterns such as `no space left on device`.
   - Run safe cleanup in this order:
     - `docker image prune -f`
     - `docker builder prune -f`
     - `docker container prune -f`
     - `docker volume prune -f`
     - `docker system prune -f` (only if still blocked)
   - Retry build once.
4. If still failing, return a clear failure report with exact command error and recommended next action.

## Safety Rules
- Never remove tagged images that are still in active use unless explicitly requested.
- Prefer targeted prune commands before broad `docker system prune`.
- Do not run destructive host commands outside Docker unless explicitly requested.

## Optional Deep Cleanup (Linux workaround)
Use only if user explicitly approves and standard prune does not recover space:
1. Stop running containers.
2. Stop Docker service.
3. Move Docker data directory from `/var/lib/docker` to `/mnt/docker`.
4. Create a symlink from `/var/lib/docker` to `/mnt/docker`.
5. Restart Docker and verify with `docker ps` and `docker images`.

## Output
Return:
1. Build status (`success` or `failed`)
2. Built image reference (`name:tag`) when successful
3. Cleanup actions performed
4. Remaining blockers (if any)
5. Exact commands the user can run for manual cleanup if unresolved
