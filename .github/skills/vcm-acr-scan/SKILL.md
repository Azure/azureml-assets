---
name: vcm-acr-scan
description: "Scan ACR-built Docker images for vulnerabilities using VCM. Use when: vcm scan, acr scan, vulnerability scan acr, vcm vulnerabilities show, vcm vulnerabilities evaluate, scan remote image, scan registry image, scan built image, vulnerability report, compare scans"
---

# VCM ACR Scan

Scans Docker images built on Azure Container Registry (ACR) for vulnerabilities using the VCM CLI (`vcm image vulnerabilities show` and `vcm image vulnerabilities evaluate`). Unlike the local `image-scanning` skill, this scans images directly on ACR — no local Docker Desktop or Trivy required.

## When to Use

- Scan images that were built on ACR via `vcm image build`
- Get vulnerability reports for remote registry images
- Evaluate compliance posture of ACR-hosted images
- Compare pre-fix vs post-fix vulnerability scans
- Batch-scan multiple images after a round of builds

## Prerequisites

- Virtual environment active with VCM installed:
  ```powershell
  .venv\Scripts\Activate.ps1
  uv pip install vcm -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/
  ```
- Azure CLI logged in (`az login`)
- Images already built on ACR with `--generate-sbom` flag (see `vcm-acr-build` skill)
- A `vuln_reports/` directory for saving scan outputs:
  ```powershell
  New-Item -ItemType Directory -Path vuln_reports -Force
  ```

## Step-by-Step Procedure

### Step 1 — Show vulnerabilities for a single image

```powershell
vcm image vulnerabilities show `
    --registry <acr-name> `
    --repository azureml/<image-name> `
    --tag <tag> `
    --output vuln_reports\<image-name>-vulns.json
```

**Parameters:**

| Parameter | Description | Example |
|-----------|-------------|---------|
| `--registry` | ACR registry name (without `.azurecr.io`) | `myteamacr` |
| `--repository` | Image repository path | `azureml/acft-grpo` |
| `--tag` | Image tag to scan | `pr-test-v1` |
| `--output` | File path to save vulnerability report (JSON) | `vuln_reports/acft-grpo-vulns.json` |

The output is a JSON file containing vulnerability details from the CoMET API, including:
- Vulnerability IDs (QID, CVE)
- Package name and installed version
- Severity/risk level
- Due dates for SLA compliance
- Fix availability and recommended versions

### Step 2 — Evaluate compliance

```powershell
vcm image vulnerabilities evaluate `
    --registry <acr-name> `
    --repository azureml/<image-name> `
    --tag <tag>
```

Returns pass/fail compliance evaluation. The image is **clean** when no actionable HIGH/CRITICAL findings remain.

#### Compliance overrides

```powershell
# Tighten the SLA window
vcm image vulnerabilities evaluate --registry <acr-name> --repository azureml/<image-name> --tag <tag> --override evaluation.sla=-30

# Ignore LOW and MEDIUM
vcm image vulnerabilities evaluate --registry <acr-name> --repository azureml/<image-name> --tag <tag> --override evaluation.ignore_risk=LOW,MEDIUM

# Suppress specific QIDs (accepted risk / exceptions filed)
vcm image vulnerabilities evaluate --registry <acr-name> --repository azureml/<image-name> --tag <tag> --override evaluation.ignore_qid=123456,789012
```

### Step 3 — Parse and analyze the vulnerability report

The JSON output can be parsed to extract actionable findings:

```python
import json

with open('vuln_reports/<image-name>-vulns.json') as f:
    data = json.load(f)

# Count by severity
from collections import Counter
risks = Counter()
for vuln in data.get('vulnerabilities', []):
    risks[vuln.get('risk', 'UNKNOWN')] += 1
print(f"Findings: {dict(risks)}")

# List HIGH/CRITICAL with package info
for vuln in data.get('vulnerabilities', []):
    if vuln.get('risk') in ('HIGH', 'CRITICAL'):
        print(f"  {vuln.get('vulnerabilityId')}: {vuln.get('vulnerabilityName')} "
              f"in {vuln.get('packagePath', 'unknown')} — risk={vuln.get('risk')}")
```

### Step 4 — Cross-reference with Kusto data

After scanning, compare VCM findings against the Kusto vulnerability list:

| Scenario | Action |
|----------|--------|
| Kusto CVE gone in VCM scan | ✅ Fix confirmed — mark as resolved |
| Kusto CVE still in VCM scan | ❌ Fix not effective — return to Dockerfile patching |
| New CVE in VCM scan (not in Kusto) | ⚠️ New finding — investigate and fix if HIGH/CRITICAL |
| Kusto CVE not found in VCM | Likely fixed by transitive upgrade — confirm with `pip show` |

## Comparing Pre-Fix vs Post-Fix Scans

Use different tags to track before/after:

```powershell
# Pre-fix scan (tag: pr-test-v1)
vcm image vulnerabilities show --registry <acr-name> --repository azureml/acft-grpo --tag pr-test-v1 --output vuln_reports\acft-grpo-vulns-v1.json

# Post-fix scan (tag: pr-test-v2)
vcm image vulnerabilities show --registry <acr-name> --repository azureml/acft-grpo --tag pr-test-v2 --output vuln_reports\acft-grpo-vulns-v2.json
```

Compare with a script:

```python
import json

def load_vulns(path):
    with open(path) as f:
        data = json.load(f)
    return {v['vulnerabilityId']: v for v in data.get('vulnerabilities', [])}

v1 = load_vulns('vuln_reports/acft-grpo-vulns-v1.json')
v2 = load_vulns('vuln_reports/acft-grpo-vulns-v2.json')

fixed = set(v1.keys()) - set(v2.keys())
new = set(v2.keys()) - set(v1.keys())
remaining = set(v1.keys()) & set(v2.keys())

print(f"Fixed: {len(fixed)}, New: {len(new)}, Remaining: {len(remaining)}")
for vid in fixed:
    print(f"  ✅ FIXED: {vid} — {v1[vid].get('vulnerabilityName', '')}")
for vid in new:
    print(f"  ⚠️  NEW: {vid} — {v2[vid].get('vulnerabilityName', '')}")
```

## Batching Multiple Image Scans

Scan all images in a batch after builds complete:

```powershell
$registry = "<acr-name>"
$tag = "pr-test-v1"
$images = @(
    "acft-grpo",
    "acft-hf-nlp-gpu",
    "acft-rft-training",
    "acft-medimageinsight-adapter-finetune",
    "acft-medimageinsight-embedding-generator",
    "acft-medimageinsight-embedding",
    "acft-medimageparse-finetune",
    "acft-mmdetection-image-gpu",
    "acft-mmtracking-video-gpu",
    "acft-multimodal-gpu",
    "acft-transformers-image-gpu",
    "acpt-automl-image-framework-selector-gpu",
    "acpt-pytorch-2.2-cuda12.1",
    "acpt-pytorch-2.8-cuda12.6",
    "ai-ml-automl-dnn-text-gpu",
    "ai-ml-automl-dnn-text-gpu-ptca",
    "ai-ml-automl-dnn-vision-gpu",
    "tensorflow-2.16-cuda11",
    "tensorflow-2.16-cuda12"
)

New-Item -ItemType Directory -Path vuln_reports -Force | Out-Null

foreach ($img in $images) {
    Write-Output "`n=== Scanning $img ==="
    try {
        vcm image vulnerabilities show `
            --registry $registry `
            --repository "azureml/$img" `
            --tag $tag `
            --output "vuln_reports\$img-vulns.json"
        Write-Output "Scan saved: vuln_reports\$img-vulns.json"
    } catch {
        Write-Output "FAILED to scan $img : $_"
    }
}
Write-Output "`nAll scans complete."
```

## Generating a Summary Report

After scanning all images, generate a consolidated report:

```python
import json, os
from collections import Counter

report = []
vuln_dir = 'vuln_reports'

for fname in sorted(os.listdir(vuln_dir)):
    if not fname.endswith('-vulns.json'):
        continue
    image = fname.replace('-vulns.json', '')
    with open(os.path.join(vuln_dir, fname)) as f:
        data = json.load(f)
    vulns = data.get('vulnerabilities', [])
    risks = Counter(v.get('risk', 'UNKNOWN') for v in vulns)
    report.append({
        'image': image,
        'total': len(vulns),
        'critical': risks.get('CRITICAL', 0),
        'high': risks.get('HIGH', 0),
        'medium': risks.get('MEDIUM', 0),
        'low': risks.get('LOW', 0),
    })

# Print summary table
print(f"{'Image':<50} {'Total':>5} {'CRIT':>5} {'HIGH':>5} {'MED':>5} {'LOW':>5}")
print("-" * 75)
for r in report:
    print(f"{r['image']:<50} {r['total']:>5} {r['critical']:>5} {r['high']:>5} {r['medium']:>5} {r['low']:>5}")
```

## Typical Durations

| Operation | Typical Duration |
|-----------|-----------------|
| `vcm image vulnerabilities show` | 10–120 seconds |
| `vcm image vulnerabilities evaluate` | 10–60 seconds |
| Full batch scan (19 images) | 15–40 minutes |

## Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `vcm: command not found` | VCM not installed in active venv | `uv pip install vcm -i https://msdata.pkgs.visualstudio.com/_packaging/Vienna/pypi/simple/` |
| Scan returns empty or no vulnerabilities | SBOM not generated during build | Rebuild with `--generate-sbom` flag, or let VCM auto-generate (adds ~5-10 min) |
| `404` or `not found` | Image/tag doesn't exist in registry | Verify with `az acr repository show-tags --name <acr> --repository azureml/<image>` |
| `unauthorized` or `403` | Not logged in or no ACR access | Run `az login` and `az acr login --name <acr-name>` |
| Scan hangs or times out | CoMET API issue | Wait 30s and retry; check VPN connection |
| JSON output is malformed | VCM returned error text instead of JSON | Check stderr output; the `--output` file may contain error messages instead of scan data |
| Scan shows vulns that should be fixed | Fix not included in the built image | Verify the build context has the fix: check the Dockerfile/requirements in `build_contexts/` |
| Windows cp1252 encoding crash | Unicode chars in scan output | Set `$env:PYTHONUTF8 = "1"` before running `vcm` commands |

## Important Notes

### Windows encoding (PYTHONUTF8)
On Windows, always set `$env:PYTHONUTF8 = "1"` before running VCM commands. Scan output may contain Unicode characters that crash the default Windows cp1252 encoding:
```powershell
$env:PYTHONUTF8 = "1"
```

### SBOM auto-generation
If the image was built without `--generate-sbom` (or SBOM generation failed), `vcm image vulnerabilities show` will automatically:
1. Pull the image from ACR as an ACR task
2. Run Trivy to generate the SBOM
3. Attach the SBOM to the image manifest
4. Then scan the SBOM via CoMET API

This adds ~5-10 minutes but means you don't need to rebuild just for a missing SBOM.

### Scan result JSON structure
The output JSON has this structure:
```json
{
  "_version": "...",
  "scanId": "...",
  "vulnerabilities": [
    {
      "vulnerabilityId": "...",
      "vulnerabilityName": "CVE-...",
      "risk": "HIGH",
      "packagePath": "...",
      ...
    }
  ]
}
```
Access `data['vulnerabilities']` to iterate over findings. An empty `vulnerabilities` array means the image is **clean**.
