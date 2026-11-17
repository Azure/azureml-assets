This model is an optimized ONNX export of NVIDIA Nemotron 3.5 Bolt 74B-A9B 24GB for local text inference on CUDA GPUs.

# ONNX Models
The CUDA variant is cataloged with RTN quantization; its source configuration declares ModelOpt mixed precision. The CUDA export includes packed expert weights, Mamba-2 recurrent states, attention caches, shared embedding initializers, and a DFlash2 speculative decoder. The main decoder uses non-paged attention caches; the drafter uses paged attention.

# Model Description
- **Developed by:** NVIDIA Corporation
- **Base model:** nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
- **Model type:** ONNX
- **License:** OpenMDW-1.1 (custom).
- **Context length:** Up to 256K tokens (`K = 1024`), matching the exported runtime configuration.
- **Capabilities:** Coding, reasoning, and tool calling.
- **Protocol:** Reasoning uses `<think>` and `</think>`. Tool calls use `<tool_call>` with nested `<function=...>` and `<parameter=...>` XML-style arguments, rather than JSON tool-call bodies.
- **Memory requirements:** Approximately 34 GiB VRAM measured for one active request on an NVIDIA A100. Requirements vary by hardware, workload, and concurrency.
- **Modalities:** This export supports text only; the upstream vision encoder is not included.
- **Disclaimer:** This model is an optimization of the base model. Please verify and test it for your scenarios; optimizations can introduce differences in output.

# Base Model Information
See the Hugging Face model [NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB) for the upstream architecture, capabilities, benchmarks, and [OpenMDW-1.1 license](https://openmdw.ai/license/1-1/).
