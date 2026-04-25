---
name: add-inspectai-benchmark
description: Add a new inspect_ai benchmark specification and dataset to the Azure ML assets ecosystem. Guides through creating benchmarkspec, dataset asset, blob storage upload, and registry configuration.
---

# Adding inspect_ai Benchmarks to Azure ML Assets

This skill guides you through adding a new inspect_ai-based benchmark to the Azure ML assets repository.

## Overview

Each inspect_ai benchmark requires:
1. **Benchmarkspec** — defines the benchmark metadata, evaluator config, and dataset reference
2. **Dataset asset** — defines the data schema, storage location, and licensing
3. **Blob storage upload** — the actual data files in Azure blob storage
4. **Registry configs** — Azure DevOps PRs for azureml/azureml-staging/azureml-dev registries (separate repo)

## Step-by-Step Process

### 1. Research the benchmark

Before creating files, gather:
- **inspect_ai task name**: e.g., `inspect_evals/browse_comp` — find in the `@task` decorator in the inspect_evals source
- **Data source**: Where does the task load data from? (HuggingFace, URL, bundled file)
- **Data format**: CSV, JSONL, or other
- **Record count**: Number of samples in the evaluation set
- **License**: Check the source repository's LICENSE file
- **Paper**: The academic paper URL
- **Column schema**: What fields does the data have?

The inspect_evals package is typically installed at:
`C:\Users\ilmat\AppData\Local\miniconda3\Lib\site-packages\inspect_evals\`

### 2. Export and upload data

Export the dataset to a file (CSV or JSONL) and upload to blob storage:

```bash
# Upload data file
az storage blob upload \
  --account-name foundrybenchmarkdatasets \
  --container-name data \
  --file "local_data_file.jsonl" \
  --name "foundry_benchmark_dataset_assets/{name}/v1/{name}.jsonl" \
  --overwrite --auth-mode login

# Upload license file
az storage blob upload \
  --account-name foundrybenchmarkdatasets \
  --container-name data \
  --file "license.txt" \
  --name "foundry_benchmark_dataset_assets/{name}/v1/license.txt" \
  --overwrite --auth-mode login
```

### 3. Create benchmarkspec

Create directory: `assets/benchmarkspecs/builtin/inspect_ai_{name}/`

**asset.yaml:**
```yaml
type: benchmarkspec
spec: spec.yaml
categories:
- BenchmarkSpec
```

**spec.yaml** (template):
```yaml
type: "benchmarkspec"
name: "builtin.inspect_ai.{name}"
version: 1
display_name: "{Display Name} Benchmark (inspect_ai)"
description: "{Description}. Evaluated using the inspect_ai framework."
benchmarkType: "builtin"
categories: ["reasoning", "quality"]

evaluator:
  testingCriteria:
    type: "inspect_ai"
    name: "{Display Name}"
    task_name: "inspect_evals/{task_name}"

dataset:
  datasetName: "{name}"
  datasetType: "oss"
  recordCount: "{count}"
  license: "{license}"
  properties:
    domain: "{domain}"
    source_paper: "{paper_url}"
    dataset_source: "{dataset_source_url}"
  source:
    provider: "mlregistry"
    sourceDatasetId: "azureml://registries/azureml/data/{name}/versions/1"
    sourceFormat: "{jsonl|csv}"
    properties:
      file_name: "{name}.{jsonl|csv}"
      type: "uri_folder"
```

Key fields:
- `dataset_source` in properties is **required** (shows external link icon in Foundry UI)
- `sourceFormat` must match the actual file format (jsonl or csv)
- `categories` should use existing values: reasoning, quality, math, science, truthfulness, instruction_following

### 4. Create dataset asset

For **JSONL** datasets: `assets/pipelines/data/global_dataset/jsonl/{name}/`
For **CSV** datasets: `assets/pipelines/data/global_dataset/generic_csv/{name}/`

**asset.yaml:**
```yaml
name: {name}
version: 1
type: data
spec: spec.yaml
extra_config: storage.yaml
categories:
  - pipeline
```

**spec.yaml:**
```yaml
$schema: https://azuremlschemas.azureedge.net/latest/data.schema.json
name: "{{asset.name}}"
version: "{{asset.version}}"
tags:
  azureml.Designer: false
description: "{Description}"
type: uri_folder
path: sample_data/
```

**storage.yaml:**
```yaml
path:
  container_name: data
  container_path: foundry_benchmark_dataset_assets/{name}/v1
  storage_name: foundrybenchmarkdatasets
  type: azureblob
```

**sample_data/_meta.yaml:**
```yaml
type: DataFrameDirectory
extension: {}
format: {JSONL|CSV}
data: {name}.{jsonl|csv}
schema: schema/_schema.json
```

**sample_data/license.txt:**
```
{Benchmark Name} - {Brief description}

Source: {source_url}
Paper: {paper_url}

Licensed under {License} License.
```

**sample_data/schema/_schema.json:**
```json
{
  "columnAttributes": [
    {
      "name": "{column_name}",
      "type": "String",
      "isFeature": true,
      "elementType": {
        "typeName": "str",
        "isNullable": false
      }
    }
  ]
}
```

### 5. Version bumping

When fixing issues or re-releasing:
- Bump `version` in dataset `asset.yaml`
- Bump `version` in benchmarkspec `spec.yaml`
- Update `sourceDatasetId` version in benchmarkspec to match new dataset version

### 6. Release patterns

After PR merges, release using patterns like:
```
(benchmarkspec/builtin.inspect_ai.{name}|data/{name})/.+
```

### 7. Registry configs (Azure DevOps)

After the GitHub PR merges, create PRs in the `azureml-asset` Azure DevOps repo to add registry configs for:
- azureml (production)
- azureml-staging
- azureml-dev

## Existing Benchmarks Reference

| Benchmark | Task Name | Format | Path |
|-----------|-----------|--------|------|
| AIME 2025 | inspect_evals/aime2025 | JSONL | jsonl/aime2025 |
| ChemBench | inspect_evals/chembench | JSONL | jsonl/chembench |
| BrowseComp | inspect_evals/browse_comp | CSV | generic_csv/browse_comp |
| τ²-Bench Telecom | inspect_evals/tau2_telecom | JSONL | jsonl/tau2_telecom |

## Common Issues

- **Missing link icon**: Ensure `dataset_source` is set in benchmarkspec `dataset.properties`
- **Replication failure**: Bump versions on both dataset and benchmarkspec to trigger re-release
- **Wrong license**: Verify license against actual source repo (HuggingFace metadata can be wrong)
