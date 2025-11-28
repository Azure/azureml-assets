import torch


def print_gpu_info():
    print("\n=== GPU Information (PyTorch) ===")

    # Print PyTorch version
    print(f"PyTorch Version: {torch.__version__}")

    print(f"CUDA Available: {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        return

    print(f"CUDA Version (PyTorch built with): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current Device ID: {torch.cuda.current_device()}")
    print(f"Current Device Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} Details ---")
        props = torch.cuda.get_device_properties(i)
        print(f"Name: {props}")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")


if __name__ == "__main__":
    print_gpu_info()
