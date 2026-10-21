# BGE Small English v1.5 (quantized ONNX)

This asset contains Qdrant's quantized ONNX export of
[BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5).
It produces normalized, 384-dimensional English text embeddings on CPU, with
inputs truncated to 512 tokens.

These are the same unmodified model and tokenizer files selected by FastEmbed
0.8.0 for `TextEmbedding(model_name="BAAI/bge-small-en-v1.5")`. This is a raw ONNX
custom model, not an MLflow package or a Foundry Local model.

## Artifact provenance

- Source: [qdrant/bge-small-en-v1.5-onnx-q](https://huggingface.co/qdrant/bge-small-en-v1.5-onnx-q/tree/aa8f8b060edb00e03bfdd08813a2949946c8ba55).
- Source revision: `aa8f8b060edb00e03bfdd08813a2949946c8ba55`.
- ONNX file: `model_optimized.onnx`, 66,465,124 bytes.
- Runtime files total: 67,179,163 bytes, excluding the license and manifest.
- License: MIT; the upstream license is included as `LICENSE`.

`artifact-manifest.json` records the size and SHA-256 of every runtime file.
The publishing source is the dedicated `models` container in the AME account
`foundrybenchmarkdata`. Artifacts uploaded to the CORP account
`foundrybenchmarkdatasets` must be replicated to that AME container before
publishing, separate from the benchmark datasets in `data`.
The versioned blob prefix includes the source revision; existing artifacts
must not be overwritten. Only the asset definitions, license, and manifest
are stored in this repository.

## Offline use

Download a pinned registry version during the application build and verify its
runtime files against `artifact-manifest.json`. Registry downloads can preserve
the source directory name, so locate the directory containing
`model_optimized.onnx` and its tokenizer files rather than assuming a flat layout:

```python
from pathlib import Path

from fastembed import TextEmbedding

download_root = Path("/app/models/bge-small-en-v1.5-onnx-q")
model_files = list(download_root.rglob("model_optimized.onnx"))
if len(model_files) != 1:
    raise RuntimeError(f"Expected one model_optimized.onnx under {download_root}")
model_dir = model_files[0].parent

model = TextEmbedding(
    model_name="BAAI/bge-small-en-v1.5",
    specific_model_path=str(model_dir),
    local_files_only=True,
    providers=["CPUExecutionProvider"],
)
embeddings = list(model.embed(["Example text"]))
```

The explicit local directory avoids Hugging Face downloads at runtime. Publishing
this asset alone does not change existing consumers; their build and loading
configuration must be updated separately.
