#!/usr/bin/env python3
"""
Simplified Model Import using snapshot_download.
Downloads HuggingFace models directly to output directory with disk space management.
"""

import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from datetime import datetime


def get_disk_usage(path):
    """Get disk usage statistics for a given path."""
    try:
        stat = shutil.disk_usage(path)
        return {
            'total_gb': round(stat.total / (1024**3), 2),
            'used_gb': round(stat.used / (1024**3), 2),
            'free_gb': round(stat.free / (1024**3), 2),
            'free_percent': round((stat.free / stat.total) * 100, 2)
        }
    except Exception as e:
        return {'error': str(e)}


def check_all_mounts():
    """Check all mounted filesystems (Linux)."""
    mounts = []
    try:
        with open('/proc/mounts', 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    mount_point = parts[1]
                    # Skip system mounts
                    if mount_point.startswith(('/sys', '/proc', '/dev', '/run')):
                        continue
                    if os.path.exists(mount_point) and os.path.ismount(mount_point):
                        usage = get_disk_usage(mount_point)
                        if 'error' not in usage:
                            mounts.append({
                                'mount_point': mount_point,
                                'usage': usage
                            })
    except Exception as e:
        print(f"Warning: Could not read /proc/mounts: {e}", file=sys.stderr)

    # Sort by free space (descending)
    mounts.sort(key=lambda x: x['usage'].get('free_gb', 0), reverse=True)
    return mounts


def find_best_mount_for_download(min_required_gb=10):
    """Find the best mount point with sufficient space."""
    mounts = check_all_mounts()

    print("\n Available Mount Points:")
    print(f"{'Mount Point':<40} {'Free (GB)':<12} {'Total (GB)':<12}")
    print("-" * 80)

    for mount in mounts:
        mp = mount['mount_point']
        u = mount['usage']
        print(f"{mp:<40} {u['free_gb']:<12} {u['total_gb']:<12}")

    # Find best mount with sufficient space
    for mount in mounts:
        if mount['usage']['free_gb'] >= min_required_gb:
            print(f"\n Selected mount: {mount['mount_point']} ({mount['usage']['free_gb']} GB free)")
            return mount['mount_point']

    print(f"\n  No mount found with {min_required_gb}+ GB free space")
    return None


def download_hf_model(model_id, output_dir):
    """Download HuggingFace model using snapshot_download."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("ERROR: huggingface_hub not installed.", file=sys.stderr)
        print("Install with: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    # Check disk space before download
    usage = get_disk_usage(output_dir)
    print(f"\n Output directory disk space:")
    print(f"   Path: {output_dir}")
    print(f"   Free: {usage.get('free_gb', 0)} GB")
    print(f"   Total: {usage.get('total_gb', 0)} GB")

    if usage.get('free_gb', 0) < 10:
        print(f"\n WARNING: Low disk space at output_dir ({usage.get('free_gb', 0)} GB)")
        print(f"   Searching for alternative mount point with 70+ GB...")

        # Find better mount point for cache
        best_mount = find_best_mount_for_download(min_required_gb=70)
        if best_mount:
            cache_dir = os.path.join(best_mount, 'huggingface_cache')
            print(f"   Using cache at: {cache_dir}")
        else:
            cache_dir = '/tmp/huggingface_cache'
            print(f"Proceeding with /tmp/huggingface_cache (may fail if insufficient space)")
    else:
        cache_dir = '/tmp/huggingface_cache'

    local_dir = output_dir

    try:
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(local_dir, exist_ok=True)

        print(f"\n Starting download of {model_id}...")
        print(f"   Cache: {cache_dir}")
        print(f"   Local: {local_dir}")
        print("-" * 80)

        # Download model using snapshot_download
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )

        print("-" * 80)
        print(f"\n Successfully downloaded {model_id}")
        print(f" Model location: {local_dir}")
        print(f" Cache location: {cache_dir}")

        return {
            'success': True,
            'model_id': model_id,
            'local_dir': local_dir,
            'cache_dir': cache_dir,
            'timestamp': datetime.now().isoformat()
        }

    except Exception as e:
        print(f"\n Download failed: {str(e)}", file=sys.stderr)
        return {
            'success': False,
            'model_id': model_id,
            'error': str(e),
            'local_dir': local_dir,
            'cache_dir': cache_dir,
            'timestamp': datetime.now().isoformat()
        }


def main():
    parser = argparse.ArgumentParser(
        description='Simplified Model Import using snapshot_download'
    )

    # Match reference component inputs
    parser.add_argument(
        '--huggingface_id',
        type=str,
        default=None,
        help='HuggingFace model ID to download (e.g., Qwen/Qwen2.5-32B-Instruct)'
    )

    parser.add_argument(
        '--pytorch_model_path',
        type=str,
        default=None,
        help='PyTorch model path (optional, for compatibility)'
    )

    parser.add_argument(
        '--mlflow_model_path',
        type=str,
        default=None,
        help='MLflow model path (optional, for compatibility)'
    )

    parser.add_argument(
        '--task_name',
        type=str,
        default='ChatCompletion',
        help='Task name'
    )

    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory where model will be downloaded'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("SIMPLIFIED MODEL IMPORT")
    print("=" * 80)
    print(f"Model ID: {args.huggingface_id}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Cache Directory: /tmp/huggingface_cache")
    print(f"Task: {args.task_name}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    # Validate inputs - need at least one model source
    if args.huggingface_id is None and args.pytorch_model_path is None and args.mlflow_model_path is None:
        print("ERROR: Must provide one of: --huggingface_id, --pytorch_model_path, or --mlflow_model_path", file=sys.stderr)
        sys.exit(1)

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Handle different input sources
    if args.pytorch_model_path or args.mlflow_model_path:
        # Copy from existing model path to output
        source_path = args.pytorch_model_path or args.mlflow_model_path

        # Find nested config.json to determine actual model directory
        if args.pytorch_model_path:
            config_paths = list(Path(args.pytorch_model_path).rglob('config.json'))
            if config_paths:
                # Use the directory containing config.json as source
                source_path = str(config_paths[0].parent)
                print(f"Found model artifact at: {source_path}")
            else:
                print(f"Warning: No config.json found, using root path: {source_path}")

        print(f"\n Copying model from {source_path} to {args.output_dir}...")

        try:
            import shutil
            # Copy all files from source to output
            for item in Path(source_path).iterdir():
                if item.is_file():
                    shutil.copy2(item, args.output_dir)
                elif item.is_dir():
                    shutil.copytree(item, Path(args.output_dir) / item.name, dirs_exist_ok=True)

            result = {
                'success': True,
                'model_id': args.pytorch_model_path or args.mlflow_model_path,
                'source': 'pytorch_model_path' if args.pytorch_model_path else 'mlflow_model_path',
                'local_dir': args.output_dir,
                'cache_dir': '/tmp/huggingface_cache',
                'timestamp': datetime.now().isoformat()
            }
            print(f"Successfully copied model to {args.output_dir}")
        except Exception as e:
            print(f" Failed to copy model: {e}", file=sys.stderr)
            result = {
                'success': False,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    elif args.huggingface_id:
        # Download from HuggingFace
        result = download_hf_model(
            model_id=args.huggingface_id,
            output_dir=args.output_dir
        )

    # Save result metadata
    metadata_file = Path(args.output_dir) / 'model_import_metadata.json'
    with open(metadata_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nMetadata saved to: {metadata_file}")

    # Exit with appropriate code
    if result['success']:
        print("\nModel import complete!")
        sys.exit(0)
    else:
        print("\nModel import failed!")
        sys.exit(1)


if __name__ == '__main__':
    main()
