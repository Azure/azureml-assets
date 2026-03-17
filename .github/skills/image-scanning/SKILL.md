---
name: image-scanning
description: "Scan locally-built Docker images for vulnerabilities using Trivy and VCM. Use when: scan image, trivy scan, vcm scan, generate SBOM, vulnerability scan, image vulnerabilities, SBOM generate, vcm evaluate, vcm show, compliance evaluate, offline scan, local SBOM scan"
---

# Image Scanning

Scans a locally-built Docker image for vulnerabilities using Trivy (SBOM generation) and VCM (CoMET API evaluation). No registry access required — all scanning runs against local images and local SBOM files.

## When to Use

- Confirm whether a locally-built image contains the vulnerabilities reported in Kusto
- Generate a Software Bill of Materials (SBOM) for a local image
- Evaluate compliance posture before or after applying Dockerfile fixes
- Cross-reference local scan findings against the Kusto vulnerability list

## Prerequisites

- Docker image already built and present locally (`docker images <image-name>`)
- Virtual environment active with scanning tools installed:
  ```powershell
  .venv\Scripts\Activate.ps1
  # Trivy — install via winget or Chocolatey if not already present
  winget install aquasecurity.trivy
  # VCM — install from the Vienna feed
  uv pip install vcm -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/
  ```
- Trivy database up to date (or use `--skip-db-update` for offline/air-gapped scans)

## Step-by-Step Procedure

### Step 1 — Generate an SPDX SBOM with Trivy

```powershell
trivy image `
    --scanners vuln `
    --no-progress `
    --format spdx-json `
    --skip-db-update `
    --skip-java-db-update `
    --offline-scan `
    --output <image-name>-sbom.json `
    --timeout 10m0s `
    <image-name>
```

**Flag reference:**

| Flag | Purpose |
|------|---------|
| `--scanners vuln` | Vulnerability scanning only; disables slow secret scanning |
| `--no-progress` | Suppresses the animated progress bar (cleaner logs) |
| `--format spdx-json` | Produces a structured SBOM file consumable by VCM |
| `--skip-db-update` | Uses cached Trivy DB; required for offline/disconnected environments |
| `--skip-java-db-update` | Skips the Java vulnerability DB update |
| `--offline-scan` | Prevents Trivy from making any network calls during the scan |
| `--timeout 10m0s` | Prevents the scan from hanging indefinitely on large GPU images |

> **Tip**: For a quick human-readable findings summary only (no SBOM needed), use the table format instead:
> ```powershell
> trivy image --scanners vuln --severity HIGH,CRITICAL <image-name>
> ```

### Step 2 — Show vulnerabilities from the SBOM

```powershell
vcm image vulnerabilities show --sbom <image-name>-sbom.json
```

This queries the CoMET API with the SBOM data and returns a structured list of known vulnerabilities with their severity and fix availability.

### Step 3 — Evaluate compliance

```powershell
vcm image vulnerabilities evaluate --sbom <image-name>-sbom.json
```

Returns a pass/fail compliance evaluation. The image is **clean** when this command reports no actionable HIGH/CRITICAL findings that match the Kusto list.

#### Compliance overrides

```powershell
# Tighten the SLA window (flag issues expiring within 30 days)
vcm image vulnerabilities evaluate --sbom <image-name>-sbom.json --override evaluation.sla=-30

# Ignore LOW and MEDIUM (focus only on HIGH/CRITICAL)
vcm image vulnerabilities evaluate --sbom <image-name>-sbom.json --override evaluation.ignore_risk=LOW,MEDIUM

# Suppress specific QIDs that are accepted risk or have exceptions
vcm image vulnerabilities evaluate --sbom <image-name>-sbom.json --override evaluation.ignore_qid=123456,789012
```

### Step 4 — Cross-reference with Kusto results

After running VCM, compare findings against the Kusto data loaded in `.cache/kusto-vuln-results.tsv`:

| Scenario | What to do |
|----------|-----------|
| Kusto CVE resolved in local scan | Mark as fixed; include in session report |
| Kusto CVE still present locally | Fix is not yet effective — return to Dockerfile patching |
| New HIGH/CRITICAL found locally (not in Kusto) | Treat as a new finding; fix before re-scanning |
| Kusto CVE not found locally | Likely already patched by a transitive dep upgrade; confirm with `pip show` fallback |

## Scan Timeout Fallback

If the Trivy scan does not complete within 10 minutes, stop waiting and verify the fix directly from inside the running image:

```powershell
# Check one package
docker run --rm <image-name> pip show <package-name>

# Check multiple packages at once
docker run --rm <image-name> pip show <pkg1> <pkg2> <pkg3>

# Check all installed packages and grep for relevant ones
docker run --rm <image-name> pip list | grep -iE "<pkg1>|<pkg2>"

# For OS / apt packages
docker run --rm <image-name> dpkg -l <package-name>
```

**Verification logic:**
1. For each CVE in the Kusto list, look up the minimum fixed version from the `ScanResult` / `Solution` column.
2. Extract the `Version:` field from `pip show <package>` output.
3. If installed version ≥ fixed version → **treat as resolved** for this cycle.
4. If installed version < fixed version → fix is not effective; return to patching.

**Rules:**
- Always record which packages were verified this way and their installed versions in the session summary.
- This fallback does **not** replace a full scan — schedule a full Trivy scan asynchronously in the background and update the report when it finishes.
- If a package cannot be found via `pip show`, try `pip list` or `conda list` depending on the package manager used in the Dockerfile.

## Checking Multiple Conda Environments

ACPT/training images often have two conda environments (e.g., `ptca` and `base`). A standard `pip show` only covers the active env. Always check both:

```powershell
# List all conda environments in the image
docker run --rm <image-name> conda env list

# Check the active (default) environment
docker run --rm <image-name> pip show <package-name>

# Check the base conda environment explicitly
docker run --rm <image-name> conda run -n base pip show <package-name>
```

If the vulnerable package exists in **both** envs, both must be patched. See the [docker-vuln-fixer agent](../../agents/docker-vuln-fixer.agent.md) for the multi-env Dockerfile fix pattern.

## Concrete Example

```powershell
# 1. Activate venv
.venv\Scripts\Activate.ps1

# 2. Generate SBOM
trivy image `
    --scanners vuln `
    --no-progress `
    --format spdx-json `
    --skip-db-update `
    --skip-java-db-update `
    --offline-scan `
    --output acpt-pytorch-2.8-cuda12.6-sbom.json `
    --timeout 10m0s `
    acpt-pytorch-2.8-cuda12.6

# 3. Show findings
vcm image vulnerabilities show --sbom acpt-pytorch-2.8-cuda12.6-sbom.json

# 4. Evaluate compliance
vcm image vulnerabilities evaluate --sbom acpt-pytorch-2.8-cuda12.6-sbom.json
```

## Batching Multiple Images

Generate SBOMs sequentially (Trivy is CPU/IO-intensive and does not benefit from parallelism on a single machine):

```powershell
$images = @("image-a", "image-b", "image-c")
foreach ($img in $images) {
    trivy image --scanners vuln --no-progress --format spdx-json `
        --skip-db-update --skip-java-db-update --offline-scan `
        --output "$img-sbom.json" --timeout 10m0s $img
    vcm image vulnerabilities show --sbom "$img-sbom.json"
    vcm image vulnerabilities evaluate --sbom "$img-sbom.json"
}
```

## Typical Durations

| Operation | Typical Duration |
|-----------|-----------------|
| `trivy image` (SBOM generation, CPU image) | 2–5 minutes |
| `trivy image` (SBOM generation, GPU image) | 5–10 minutes |
| `vcm image vulnerabilities show` | 10–60 seconds |
| `vcm image vulnerabilities evaluate` | 10–60 seconds |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `trivy: command not found` | Trivy not installed or not on PATH | `winget install aquasecurity.trivy` or add to PATH |
| `FATAL image scan failed: ... timeout` | Image too large for default timeout | Increase `--timeout` to `20m0s` or `30m0s` |
| `vcm: command not found` | VCM not installed in active venv | `uv pip install vcm -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/` |
| VCM returns connection error | CoMET API unreachable | Check VPN; wait 10 s and retry once |
| `No such image` in Trivy | Image not built locally | Run the docker-local-build skill first |
| SBOM is 0 bytes or empty | Trivy scan interrupted / timed out | Re-run with longer `--timeout`; check disk space |
