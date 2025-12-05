# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import os
import sys
import json
from copy import deepcopy

from bench_serving import run_benchmark
from helper import get_api_key_from_connection, log_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="SGLang Benchmarking")
    # parser.add_argument(
    #     "--metrics_path",
    #     type=str,
    #     required=True,
    #     help="Output JSON file to store the benchmarking metrics.",
    # )
    parser.add_argument(
        "--connection-name",
        type=str,
        required=True,
        help="The name of the workspace connection used to fetch API key for base endpoint.",
    )
    parser.add_argument(
        "--target-url",
        type=str,
        required=True,
        help="Server or API base url for target endpoint.",
    )
    parser.add_argument(
        "--target-connection-name",
        type=str,
        required=True,
        help="The name of the workspace connection used to fetch API key for target endpoint.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of trials to run the benchmark, result will be averaged over all trials.",
    )
    parser.add_argument(
        "--base-backend",
        type=str,
        default="sglang",
        help="Backend for base endpoint, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--target-backend",
        type=str,
        default="sglang",
        help="Backend for target endpoint, depending on the LLM Inference Engine.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Server or API base url if not using http host and port.",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Default host is 0.0.0.0.")
    parser.add_argument(
        "--port",
        type=int,
        help="If not set, the default port is configured according to its default value for" \
        " different LLM Inference Engines.",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="sharegpt",
        choices=[
            "sharegpt",
            "random",
            "generated-shared-prefix",
            "ultrachat",
            "loogle",
            "nextqa",
        ],
        help="Name of the dataset to benchmark on.",
    )
    parser.add_argument("--dataset-path", type=str, default="", help="Path to the dataset.")
    parser.add_argument(
        "--base-model",
        type=str,
        help="Name or path of the model for base endpoint. If not set, the default model will request "
        "/v1/models for conf.",
        default=None,
    )
    parser.add_argument(
        "--target-model",
        type=str,
        help="Name or path of the model for target endpoint. If not set, the default model will request "
        "/v1/models for conf.",
        default=None,
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        help="Name or path of the tokenizer. If not set, using the model conf.",
    )
    parser.add_argument(
        "--chat-template",
        type=str,
        help="The buliltin chat template name or the path of the chat template file. This is only used " \
        "for OpenAI-compatible API server.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=1000,
        help="Number of prompts to process. Default is 1000.",
    )
    parser.add_argument(
        "--fixed-output-len",
        type=int,
        default=None,
        help="Output length for each request. Overrides the output length from the dataset.",
    )
    parser.add_argument(
        "--sharegpt-context-len",
        type=int,
        default=None,
        help="The context length of the model for the ShareGPT dataset. Requests longer than the " \
        "context length will be dropped.",
    )
    parser.add_argument(
        "--random-input-len",
        type=int,
        default=1024,
        help="Number of input tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-output-len",
        default=1024,
        type=int,
        help="Number of output tokens per request, used only for random dataset.",
    )
    parser.add_argument(
        "--random-range-ratio",
        type=float,
        default=0.0,
        help="Range of sampled ratio of input/output length, " "used only for random dataset.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=None,
        help="Maximum number of concurrent requests. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--multi",
        action="store_true",
        help="Use request rate range rather than single value.",
    )
    parser.add_argument(
        "--request-rate-range",
        type=str,
        default="2,34,2",
        help="Range of request rates in the format start,stop,step. Default is 2,34,2. It also supports " \
        "a list of request rates, requiring the parameters to not equal three.",
    )
    parser.add_argument("--output-file", type=str, help="Output JSONL file name.")
    parser.add_argument(
        "--enable-multiturn",
        action="store_true",
        help="Enable multiturn chat for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "sharegpt, ultrachat, loogle, nextqa",
    )
    parser.add_argument(
        "--enable-shared-prefix",
        action="store_true",
        help="Enable shared prefix for online serving benchmarking. "
        "This option is effective on the following datasets: "
        "loogle, nextqa",
    )
    parser.add_argument(
        "--disable-shuffle",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=False,
        help="Disable shuffling datasets. Accepts true/false.",
    )
    parser.add_argument(
        "--disable-tqdm",
        action="store_true",
        help="Specify to disable tqdm progress bar.",
    )
    parser.add_argument(
        "--disable-stream",
        action="store_true",
        help="Disable streaming mode.",
    )
    parser.add_argument(
        "--return-logprob",
        action="store_true",
        help="Return logprob.",
    )
    parser.add_argument("--seed", type=int, default=1, help="The random seed.")
    parser.add_argument(
        "--disable-ignore-eos",
        action="store_true",
        help="Disable ignoring EOS.",
    )
    parser.add_argument(
        "--extra-request-body",
        metavar='{"key1": "value1", "key2": "value2"}',
        type=str,
        help="Append given JSON object to the request payload. You can use this to specify"
        "additional generate params like sampling params.",
    )
    parser.add_argument(
        "--apply-chat-template",
        action="store_true",
        help="Apply chat template",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Use Torch Profiler. The endpoint must be launched with " "SGLANG_TORCH_PROFILER_DIR to enable profiler.",
    )
    parser.add_argument(
        "--lora-name",
        type=str,
        default=None,
        help="The name of LoRA adapter",
    )

    group = parser.add_argument_group("generated-shared-prefix dataset arguments")
    group.add_argument(
        "--gsp-num-groups",
        type=int,
        default=64,
        help="Number of system prompt groups for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-prompts-per-group",
        type=int,
        default=16,
        help="Number of prompts per system prompt group for generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-system-prompt-len",
        type=int,
        default=2048,
        help="Target length in tokens for system prompts in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-question-len",
        type=int,
        default=128,
        help="Target length in tokens for questions in generated-shared-prefix dataset",
    )
    group.add_argument(
        "--gsp-output-len",
        type=int,
        default=256,
        help="Target length in tokens for outputs in generated-shared-prefix dataset",
    )
    # videos specific
    parser.add_argument(
        "--max-frames",
        type=int,
        default=sys.maxsize,
        help="The maximum number of frames to extract from each video. "
        "This option is specific to the nextqa dataset (video benchmark). ",
    )
    args = parser.parse_args()
    return args


def _generate_avg_metrics(metrics_file: str, prefix: str = "", log_to_aml: bool = True):
    metrics_list = []
    with open(metrics_file, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                metrics_list.append(json.loads(line))

    # Compute average metrics
    avg_metrics = {}
    count = len(metrics_list)

    for key in metrics_list[0].keys():
        if isinstance(metrics_list[0][key], (int, float)):
            avg_metrics[key] = sum(result[key] for result in metrics_list) / count
        else:
            avg_metrics[key] = metrics_list[0][key]

    output_file = os.path.join(os.path.dirname(metrics_file), f"{prefix}metrics_avg.json")
    with open(output_file, "w") as f:
        json.dump(avg_metrics, f, indent=4)

    # Log metrics with prefix only if requested (not in child process)
    if log_to_aml:
        # Convert keys to uppercase for consistent bold display
        prefixed_metrics = {}
        for k, v in avg_metrics.items():
            metric_key = k.upper()
            prefixed_metrics[f"{prefix}{metric_key}"] = v
        log_metrics(prefixed_metrics)

    return avg_metrics


def run_endpoint_benchmark(
    args,
    endpoint_name: str,
    url: str,
    connection_name: str,
    model: str,
    backend: str,
    output_dir: str,
):
    """Run benchmark for a specific endpoint"""
    print(f"\n{'='*60}")
    print(f"Starting benchmark for {endpoint_name} endpoint")
    print(f"{'='*60}\n")

    # Create a copy of args for this endpoint
    endpoint_args = deepcopy(args)
    endpoint_args.base_url = url
    endpoint_args.model = model
    endpoint_args.backend = backend

    # Remove last slash if exists
    if endpoint_args.base_url and endpoint_args.base_url.endswith("/"):
        endpoint_args.base_url = endpoint_args.base_url[:-1]

    # Get API key for this endpoint
    api_key, _ = get_api_key_from_connection(connection_name)
    os.environ["OPENAI_API_KEY"] = api_key

    # Set output file for this endpoint
    endpoint_args.output_file = os.path.join(output_dir, f"{endpoint_name}_metrics_each_trial.jsonl")

    trials = endpoint_args.trials
    del endpoint_args.trials

    # Run trials
    for trial in range(trials):
        print(f"[{endpoint_name}] Starting trial {trial + 1} of {trials}...")
        try:
            run_benchmark(endpoint_args)
        except Exception as e:
            print(f"[{endpoint_name}] Trial {trial + 1} failed with error: {e}")
            import traceback

            traceback.print_exc()
            raise

    # Generate average metrics with prefix (don't log in child process)
    _generate_avg_metrics(endpoint_args.output_file, prefix=f"{endpoint_name}_", log_to_aml=False)
    print(f"\n[{endpoint_name}] Benchmark completed!\n")


def main():
    args = parse_args()
    print("> Parsed arguments:", args)

    # Store endpoint configurations
    base_url = args.base_url
    base_connection = args.connection_name
    base_model = args.base_model
    base_backend = args.base_backend
    target_url = args.target_url
    target_connection = args.target_connection_name
    target_model = args.target_model
    target_backend = args.target_backend
    output_dir = args.output_file

    # Remove endpoint-specific args from the base args
    del args.connection_name
    del args.base_model
    del args.base_backend
    del args.target_url
    del args.target_connection_name
    del args.target_model
    del args.target_backend
    del args.output_file
    del args.base_url

    # Run benchmarks sequentially
    print("\n" + "=" * 60)
    print("Starting sequential benchmarks for base and target endpoints")
    print("=" * 60 + "\n")

    # Run base endpoint benchmark first
    print("Running base endpoint benchmark...")
    try:
        run_endpoint_benchmark(
            args,
            "base",
            base_url,
            base_connection,
            base_model,
            base_backend,
            output_dir,
        )
        print("Base endpoint benchmark completed.\n")
    except Exception as e:
        print(f"[ERROR] Base endpoint benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Run target endpoint benchmark second
    print("Running target endpoint benchmark...")
    try:
        run_endpoint_benchmark(
            args,
            "target",
            target_url,
            target_connection,
            target_model,
            target_backend,
            output_dir,
        )
        print("Target endpoint benchmark completed.\n")
    except Exception as e:
        print(f"[ERROR] Target endpoint benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("All benchmarks completed successfully!")
    print("=" * 60 + "\n")

    # Log metrics to AML from main process
    print("Logging metrics to AzureML...")
    try:
        base_metrics_file = os.path.join(output_dir, "base_metrics_avg.json")
        target_metrics_file = os.path.join(output_dir, "target_metrics_avg.json")

        if os.path.exists(base_metrics_file):
            with open(base_metrics_file, "r") as f:
                base_metrics = json.load(f)
                # Create prefixed metrics with uppercase keys for bold display
                prefixed_base = {}
                for k, v in base_metrics.items():
                    metric_key = k.upper()  # Convert to uppercase
                    prefixed_base[f"base_{metric_key}"] = v
                log_metrics(prefixed_base)
                print(f"  ✓ Logged {len(base_metrics)} base metrics")

        if os.path.exists(target_metrics_file):
            with open(target_metrics_file, "r") as f:
                target_metrics = json.load(f)
                # Create prefixed metrics with uppercase keys for bold display
                prefixed_target = {}
                for k, v in target_metrics.items():
                    metric_key = k.upper()  # Convert to uppercase
                    prefixed_target[f"target_{metric_key}"] = v
                log_metrics(prefixed_target)
                print(f"  ✓ Logged {len(target_metrics)} target metrics")

        print("Metrics logging completed!")
    except Exception as e:
        print(f"Warning: Failed to log metrics to AML: {e}")


if __name__ == "__main__":
    main()
