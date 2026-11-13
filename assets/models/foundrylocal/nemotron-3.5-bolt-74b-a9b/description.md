This model is an optimized ONNX export of NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB for local text inference on CUDA GPUs.

# ONNX Models
The CUDA variant is cataloged with RTN quantization; its source configuration declares ModelOpt mixed precision. The CUDA export includes packed expert weights, Mamba-2 recurrent states, attention caches, shared embedding initializers, and a DFlash2 speculative decoder. The main decoder uses non-paged attention caches; the drafter uses paged attention.

# Model Description
- **Developed by:** NVIDIA Corporation
- **Base model:** nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
- **Model type:** ONNX
- **License:** OpenMDW-1.1 (custom).
- **Configured runtime token limit:** 135160 tokens (`132 * 1024 - 8`). The decoder's KV cache holds 135168 tokens (132 Ki tokens); the eight-token difference matches the configured speculative block size. This is an export-specific runtime limit, not a rounded advertised context window. The upstream 262144-token context is not enabled.
- **Capabilities:** Coding, reasoning, and tool calling.
- **Protocol:** Reasoning uses `<think>` and `</think>`. Tool calls use `<tool_call>` with nested `<function=...>` and `<parameter=...>` XML-style arguments, rather than JSON tool-call bodies.
- **Memory requirements:** The catalog declares 31 GiB (33,285,996,544 bytes) of VRAM. The `consumer-gpu-32g` and `gpu-132k` profiles apply identical full-context configurations with expert offloading disabled. The lower profile's minimum device-memory threshold is 32,212,254,721 bytes, rounded up to a whole GiB for the catalog; a 32 GiB GPU is the corresponding practical hardware tier. Lower-memory profiles allow adaptive expert offloading and have different host-memory requirements. This requirement is configuration-derived, not a locally measured inference peak.
- **Modalities:** This export supports text only; the upstream vision encoder is not included.
- **Disclaimer:** This model is an optimization of the base model. Please verify and test it for your scenarios; optimizations can introduce differences in output.

# Base Model Information
See the Hugging Face model [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) for the upstream architecture, capabilities, benchmarks, and [OpenMDW-1.1 license](https://openmdw.ai/license/1-1/).
