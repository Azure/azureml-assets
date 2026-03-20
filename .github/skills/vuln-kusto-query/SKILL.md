---
name: vuln-kusto-query
description: "Fetch vulnerability data from Kusto for Azure ML curated images. Use when: query vulnerabilities, check SLA status, get vulnerability report from Kusto, which images are out of SLA, approaching SLA, vulnerability status, curated image vulnerabilities, Training owner vulnerabilities"
---

# Vulnerability Kusto Query

Fetches actionable vulnerability data for Azure ML curated container images from the ShaVulnMgmt Kusto cluster. Returns images that are out of SLA or approaching SLA, filtered to the latest image tags.

## When to Use

- Check which curated images have outstanding vulnerabilities
- See which vulnerabilities are out of SLA or approaching SLA deadline
- Get vulnerability details (ID, name, status, due date) for a specific image
- Review vulnerability posture before starting Dockerfile fixes

## Prerequisites

- Access to `viennause2.kusto.windows.net` and `shavulnmgmtprdwus.kusto.windows.net` Kusto clusters
- Azure CLI logged in (`az login`)
- Kusto CLI or Kusto Explorer, or use the Azure Data Explorer web UI at https://dataexplorer.azure.com

## Query

Run this query against the `shavulnmgmtprdwus.kusto.windows.net/ShaVulnMgmt` cluster/database:

```kql
let SLA = 14d;
let IncludeDeprecated = false;
let _startTime = ago(36h);
let _endTime = now();
let _owner = dynamic(null);
let registry="azuremlmcr.azurecr.io";
let knownenvs=cluster("viennause2.kusto.windows.net").database("Vienna").curatedimages;
let exceptions=cluster("viennause2.kusto.windows.net").database("Vienna").curatedimages_exceptions;
let outOfSLA = cluster('shavulnmgmtprdwus.kusto.windows.net').database('ShaVulnMgmt').VulnDetailsContainerImage(_startTime,_endTime)
| where ServiceTreeId=='776fcc4a-80c5-455a-8490-449352e5b55b'
| where AssetType == 'Container Registry'
| where ScanAttributes["RegistryName"] ==  registry
| parse ScanAttributes["ImageName"] with Registry "/" Image ":" Tag
| where ScanAttributes["ImageTag"] matches regex @"\d{8}.v\d{1}" or (ScanAttributes["ImageTag"] matches regex @"\d" and Image matches regex "(?:public|unlisted)/azureml/curated/(.+)")
| join kind=inner knownenvs on $left.Image==$right.Image
| where Supported==true or IncludeDeprecated==true
| where (isempty(['_owner']) or Owner in (['_owner']))
| project RunId, Registry, Image, Tag, AssetId, Supported, Owner, IsActionable, VulnerabilityId, VulnerabilityName, DueDate=todatetime(VulnerabilityAttributes.DueDate_ForS360), Risk=tostring(VulnerabilityAttributes.Risk), Solution=tostring(VulnerabilityAttributes.SOLUTION), ScanResult=tostring(ScanResult);
let latestCEImages = outOfSLA
| where Tag matches regex @"\d" and Image matches regex "(?:public|unlisted)/azureml/curated/(.+)"
| summarize Tag=max(toint(Tag)) by Image
| project repo=strcat(Image, ":", Tag);
let latestBaseImages = outOfSLA
| where Tag matches regex @"\d{8}.v\d{1}" 
| summarize Tag=max(Tag) by Image
| project repo=strcat(Image, ":", Tag);
outOfSLA
| where IsActionable==True
| where strcat(Image, ":", Tag) in (latestBaseImages) or strcat(Image, ":", Tag) in (latestCEImages)
| where Owner in ('Training', 'Project_24_25')
| summarize arg_max(RunId, Registry, Image, Tag, IsActionable, VulnerabilityName, DueDate, Risk, Solution, ScanResult, Supported, Owner) by AssetId, VulnerabilityId
| join kind=leftouter exceptions on $left.VulnerabilityId==$right.VulnerabilityId
| extend DueDate=iif(isempty(ExtDate), DueDate, ExtDate)
| extend SLADate=DueDate - SLA
| extend Status = iif(SLADate - now() < 0d, "OOSLA", "Approaching SLA")
| extend SLADate=format_datetime(SLADate, "yyyy-MM-dd")
| project Owner, Image, Tag, VulnerabilityId, Status, SLADate, VulnerabilityName, ScanResult
| sort by Owner
```

## Tunable Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SLA` | `14d` | Days before due date to flag as approaching SLA |
| `IncludeDeprecated` | `false` | Include deprecated/unsupported images |
| `_startTime` | `ago(36h)` | Scan window start |
| `_endTime` | `now()` | Scan window end |
| `_owner` | `dynamic(null)` | Filter by owner (null = all owners) |
| `Owner in (...)` | `'Training', 'Project_24_25'` | Final owner filter — adjust to target specific teams |

## Output Columns

| Column | Description |
|--------|-------------|
| `Owner` | Team that owns the image (e.g., Training, Project_24_25) |
| `Image` | Full image path (e.g., `public/azureml/curated/acft-hf-nlp-gpu`) |
| `Tag` | Image tag |
| `VulnerabilityId` | Unique vulnerability identifier |
| `Status` | `OOSLA` (past due) or `Approaching SLA` |
| `SLADate` | Date by which the fix must ship (due date minus SLA buffer) |
| `VulnerabilityName` | CVE or advisory name |
| `ScanResult` | Scan finding details |

## Session Caching

Kusto vulnerability data is based on scans that run daily, so results rarely change within a single working session. After running this query:
1. Save the output to `.cache/kusto-vuln-results.tsv` (tab-separated, with header row) in the workspace root.
2. On subsequent requests, read from the cache instead of re-querying Kusto.
3. To force a refresh, delete the cache file and re-run the query.
4. To filter for a specific image, read the cached file and filter locally.

This avoids repeated round-trips to the Kusto cluster and keeps the workflow fast.

## Interpreting Results

- **OOSLA**: Fix is overdue — prioritize immediately
- **Approaching SLA**: Fix needed before `SLADate` — plan the Dockerfile patch
- Cross-reference `Image` with environment folders under `assets/` to locate the Dockerfile to fix
- The `ScanResult` field often contains the package name and installed/fixed versions
