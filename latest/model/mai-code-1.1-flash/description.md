This model is an optimized ONNX export of MAI Code 1.1 Flash for local text inference on CUDA GPUs.

# ONNX Models
The CUDA export includes mixed-precision RTN weight quantization with 2-bit experts, INT4 dense projections, and an INT8 output head. It uses INT8 per-channel KV caches, paged attention, sliding-window attention, and a DFlash2 speculative decoder.

# Model Description
- **Developed by:** Microsoft
- **Model type:** ONNX
- **License:** MIT
- **Context length:** 256K tokens (nominal export context; `K = 1024`).
- **Capabilities:** Coding, reasoning, and tool calling.
- **Protocol:** Harmony-style analysis, final, tool-call, and tool-response messages, with empty reasoning and tool-call delimiter fields in the metadata. Tool definitions are rendered as a TypeScript function namespace; tool arguments are JSON.
- **Memory requirements:** Approximately 73 GiB VRAM measured for one active request on an NVIDIA A100. Requirements vary by hardware, workload, and concurrency.
- **Disclaimer:** This model is an optimization of the reference model. Please verify and test it for your scenarios; optimizations can introduce differences in output.
