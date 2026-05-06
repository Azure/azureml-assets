# Qwen3 Embedding 8B Webgpu Gpu

This is the GPU (WebGPU)-optimized variant of qwen3-embedding-8b, a text embedding model from the Qwen3 family developed by Alibaba Cloud and optimized by Microsoft.

## Model Details
- **Model Type**: Text Embedding (ONNX)
- **Parameters**: 8 billion
- **Context Length**: 32K tokens
- **Embedding Dimension**: Up to 4096
- **Quantization**: KLD Gradient quantization
- **Target Device**: GPU (WebGPU)
- **Execution Provider**: WebGPUExecutionProvider
- **Supported Languages**: 100+

## Intended Use
This model is optimized for local execution on devices with GPU (WebGPU) hardware acceleration using Foundry Local.

## Capabilities
- Text retrieval and semantic search
- Code retrieval
- Text classification and clustering
- Bitext mining
- Multilingual and cross-lingual retrieval

## License
This model is licensed under Apache 2.0. See [license details](https://huggingface.co/Qwen/Qwen3-Embedding-8B/blob/main/LICENSE).

## Source
- HuggingFace: [Qwen3-Embedding-8B](https://huggingface.co/Qwen/Qwen3-Embedding-8B)
