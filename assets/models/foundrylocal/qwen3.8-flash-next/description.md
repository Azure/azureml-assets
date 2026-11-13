This model is an optimized version of Qwen3.8-Flash-Next for local text inference on CUDA GPUs.

# ONNX Models
The CUDA package includes mixed-precision 2-, 3-, and 4-bit expert weights using RTN quantization, embedding and n-gram tables, FP16 and INT8 cache variants, and a DFlash2 speculative decoder. Hardware-specific runtime profiles select cache precision and expert offloading.

# Model Description
- **Developed by:** Microsoft
- **Base model:** Qwen/Qwen3.8-Flash-Next
- **Model type:** ONNX
- **License:** Qwen Community License Agreement 1.0 (custom).
- **Context length:** Up to 256K tokens (`K = 1024`), subject to the selected runtime profile.
- **Capabilities:** Coding, text generation, reasoning, and tool calling.
- **Protocol:** Reasoning uses `<think>` and `</think>`. Tool calls use `<tool_call>` with nested `<function=...>` and `<parameter=...>` XML-style arguments, rather than JSON tool-call bodies.
- **Memory requirements:** The catalog declares 61 GiB (65,498,251,264 bytes) of VRAM, above the export's 61 GB (61,000,000,000 bytes) minimum for the resident-expert 256K profile. The 20 GiB tier requires expert offloading and a smaller cache; the resident-expert 128K tier requires 57 GB. These are declared requirements, not locally measured inference peaks. Host-memory and disk requirements also depend on the selected profile.
- **Modalities:** This export supports text only; the upstream vision encoder is not included.
- **Disclaimer:** This model is an optimization of the base model. Please verify and test it for your scenarios; optimizations can introduce differences in output.

# Base Model Information
See the Hugging Face model [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) for the upstream architecture, capabilities, benchmarks, and [license terms](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/main/LICENSE). The native upstream context window is 256K tokens (`K = 1024`); the extended one-million-token configuration is not enabled in this export.
