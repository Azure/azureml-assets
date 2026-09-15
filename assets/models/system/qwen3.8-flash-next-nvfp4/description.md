# Qwen3.8-Flash-Next NVFP4

This custom model packages [NVIDIA's quantization of Qwen3.8-Flash-Next](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/tree/fc694b54fb0174e0913e6adf86691ef85a4ead47).
Routed experts use W4A4 NVFP4. Attention and shared experts retain BF16; PLE embeddings and MTP components use FP8. This is not uniform 4-bit quantization or the separate NVFP4A16/Marlin checkpoint.

The package requires all ten numbered safetensors shards, `model-fp8-mtp-ple.safetensors`, and the source configuration, tokenizer, and processor files. The PLE file is required even when MTP is disabled. Weight files total 132,680,249,378 bytes; runtime memory requirements differ.

## Runtime

The publisher documents Linux/vLLM on NVIDIA B200/B300. Follow the [pinned model card](https://huggingface.co/nvidia/Qwen3.8-Flash-Next-NVFP4/blob/fc694b54fb0174e0913e6adf86691ef85a4ead47/README.md) for architecture support and mixed-precision loading requirements.
This asset does not enable Azure managed inference, MaaS, batch inference, or fine-tuning. It does not claim compatibility with Foundry Local or a managed inference environment.

## License

The [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) and [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE) apply. Preserve their notices when packaging or redistributing the model. Public download availability does not waive the Qwen license's separate commercial-license conditions.
