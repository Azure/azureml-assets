#!/opt/conda/envs/ptca/bin/python
"""Validate the ACFT RFT CUDA stack and emit topology-safe NCCL settings."""

import argparse
import importlib
import importlib.metadata
import importlib.util
import os
import shlex
import sys


EXPECTED_VERSIONS = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "vllm": "0.26.0",
    "openai": "2.25.0",
    "nvidia-nccl-cu12": "2.28.9",
}
EXPECTED_CUDA = "12.9"


def normalized_version(package_name: str) -> str:
    return importlib.metadata.version(package_name).split("+", 1)[0]


def validate_versions(torch_module) -> None:
    for package_name, expected_version in EXPECTED_VERSIONS.items():
        actual_version = normalized_version(package_name)
        if actual_version != expected_version:
            raise RuntimeError(
                f"{package_name} {actual_version} does not match expected "
                f"version {expected_version}"
            )
    if torch_module.version.cuda != EXPECTED_CUDA:
        raise RuntimeError(
            f"PyTorch CUDA {torch_module.version.cuda} does not match "
            f"expected CUDA {EXPECTED_CUDA}"
        )


def unsupported_peer_pairs(torch_module) -> list[tuple[int, int]]:
    device_count = torch_module.cuda.device_count()
    return [
        (source, target)
        for source in range(device_count)
        for target in range(device_count)
        if source != target
        and not torch_module.cuda.can_device_access_peer(source, target)
    ]


def topology_exports(torch_module) -> dict[str, str]:
    if not unsupported_peer_pairs(torch_module):
        return {}

    exports = {}
    if "NCCL_P2P_DISABLE" not in os.environ:
        exports["NCCL_P2P_DISABLE"] = "1"
    if "NCCL_SHM_DISABLE" not in os.environ:
        exports["NCCL_SHM_DISABLE"] = "1"
    return exports


def attention_exports() -> dict[str, str]:
    flash_attn = importlib.import_module(
        "vllm.vllm_flash_attn.flash_attn_interface"
    )
    if flash_attn.FA2_AVAILABLE or flash_attn.FA3_AVAILABLE:
        return {}
    if "VLLM_ATTENTION_BACKEND" in os.environ:
        return {}
    return {
        "VLLM_ATTENTION_BACKEND": "TRITON_ATTN",
        "VERL_ACTOR_ATTENTION_IMPLEMENTATION": "sdpa",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--shell",
        action="store_true",
        help="Print shell export commands for topology-safe NCCL defaults.",
    )
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Validate imports and native extensions without requiring a GPU.",
    )
    args = parser.parse_args()

    import torch
    import torchvision
    import vllm
    from openai.types.responses import NamespaceTool  # noqa: F401

    validate_versions(torch)
    if importlib.util.find_spec("vllm._C_stable_libtorch") is None:
        raise RuntimeError("vLLM stable native extension is not installed")
    for package_name, reason in {
        "torchaudio": "it is unused by this image",
        "torchcodec": "its vLLM dependency wheel requires CUDA 13",
    }.items():
        try:
            importlib.metadata.version(package_name)
        except importlib.metadata.PackageNotFoundError:
            pass
        else:
            raise RuntimeError(f"{package_name} must be absent because {reason}")
    print(
        f"Validated torch={torch.__version__}, torchvision={torchvision.__version__}, "
        f"vllm={vllm.__version__}, "
        f"nccl={normalized_version('nvidia-nccl-cu12')}; "
        "torchaudio is intentionally absent",
        file=sys.stderr,
        flush=True,
    )

    exports = {}
    if not args.skip_cuda:
        importlib.import_module("vllm._C_stable_libtorch")
        exports.update(attention_exports())
        os.environ.update(exports)
        importlib.import_module("vllm.lora.lora_model")
        torch.cuda.init()
        print(
            f"CUDA devices: {torch.cuda.device_count()}; "
            f"device 0: {torch.cuda.get_device_name(0)}",
            file=sys.stderr,
            flush=True,
        )
        topology = topology_exports(torch)
        exports.update(topology)
        if topology:
            print(
                "GPU peer access is incomplete; using NCCL socket transport.",
                file=sys.stderr,
                flush=True,
            )
        if exports.get("VLLM_ATTENTION_BACKEND") == "TRITON_ATTN":
            print(
                "vLLM FA2/FA3 extensions are unavailable; using Triton attention.",
                file=sys.stderr,
                flush=True,
            )

    if args.shell:
        for name, value in exports.items():
            print(f"export {name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
