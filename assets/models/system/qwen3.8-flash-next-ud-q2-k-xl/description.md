# Qwen3.8-Flash-Next UD-Q2_K_XL (2-bit)

This custom model packages the [Unsloth Dynamic UD-Q2_K_XL quantization](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/38bb39ee97821de2c9009abb7e93950eec396e66/UD-Q2_K_XL) of Qwen3.8-Flash-Next.
The 2-bit label describes a mixed-precision GGUF variant, not uniform precision across every tensor. PLE embeddings retain at least 4-bit precision.

All three `Qwen3.8-Flash-Next-UD-Q2_K_XL-0000*-of-00003.gguf` shards are required. They total 78,869,128,864 bytes. The first shard alone is not a usable model. Preserve the `UD-Q2_K_XL` directory and load its first shard with the remaining shards alongside it.

## Runtime

Use a llama.cpp-compatible runtime with `qwen4exp` architecture support. See the [publisher's guide](https://unsloth.ai/docs/models/qwen3.8-next).
This package is text-only: it does not include the separate vision projector or optional MTP weights.
Azure managed inference, MaaS, batch inference, and fine-tuning are not enabled. Foundry Local and managed inference environment compatibility are not claimed.

## License

The [Qwen Community License 1.0](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/LICENSE) applies. Preserve its notices when packaging or redistributing the model. Public download availability does not waive its separate commercial-license conditions.
