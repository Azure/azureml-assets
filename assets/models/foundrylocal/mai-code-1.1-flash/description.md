This model is an optimized ONNX export of MAI Code 1.1 Flash for local text inference on CUDA GPUs.

# ONNX Models
The CUDA export includes mixed-precision RTN weight quantization, paged attention, sliding-window attention, and a DFlash2 speculative decoder.

# Model Description
- **Developed by:** Microsoft
- **Model type:** ONNX
- **License:** MIT
- **Configured runtime token limit:** 262145 tokens (`256 * 1024 + 1`), matching the source `max_position_embeddings`, runtime `context_length`, and `max_length` settings. This is an export-specific limit; the extra token's purpose has not been confirmed.
- **Capabilities:** Coding, reasoning, and tool calling.
- **Protocol:** Harmony-style analysis, final, tool-call, and tool-response messages, with empty reasoning and tool-call delimiter fields in the metadata. Tool definitions are rendered as a TypeScript function namespace; tool arguments are JSON.
- **Memory requirements:** This asset declares an 80 GiB (85,899,345,920 bytes) VRAM requirement. It is not a locally measured minimum for this ONNX export. Memory requirements depend on context length, paged-cache allocation, and speculative decoding.
- **Disclaimer:** This model is an optimization of the reference model. Please verify and test it for your scenarios; optimizations can introduce differences in output.
