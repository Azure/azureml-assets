#!/opt/conda/envs/ptca/bin/python
"""Validate the ACFT RFT CUDA stack and emit topology-safe NCCL settings."""

import argparse
import importlib.metadata
import os
import shlex
import sys


EXPECTED_VERSIONS = {
    "torch": "2.13.0",
    "torchvision": "0.28.0",
    "vllm": "0.26.0",
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
    import vllm._C_stable_libtorch

    validate_versions(torch)
    try:
        importlib.metadata.version("torchaudio")
    except importlib.metadata.PackageNotFoundError:
        pass
    else:
        raise RuntimeError(
            "torchaudio must be absent because no torch 2.13-compatible release exists"
        )
    print(
        f"Validated torch={torch.__version__}, torchvision={torchvision.__version__}, "
        f"vllm={vllm.__version__}; torchaudio is intentionally absent",
        file=sys.stderr,
        flush=True,
    )

    exports = {}
    if not args.skip_cuda:
        torch.cuda.init()
        print(
            f"CUDA devices: {torch.cuda.device_count()}; "
            f"device 0: {torch.cuda.get_device_name(0)}",
            file=sys.stderr,
            flush=True,
        )
        exports = topology_exports(torch)
        if exports:
            print(
                "GPU peer access is incomplete; using NCCL socket transport.",
                file=sys.stderr,
                flush=True,
            )

    if args.shell:
        for name, value in exports.items():
            print(f"export {name}={shlex.quote(value)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
