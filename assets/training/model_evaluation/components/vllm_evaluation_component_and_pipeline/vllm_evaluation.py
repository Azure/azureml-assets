#!/usr/bin/env python
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import json
import os
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from vllm import LLM, SamplingParams
from verl.utils.reward_score.gsm8k import extract_solution
from math_verify import parse, verify


def load_validation_data(validation_file: str) -> List[Dict[str, Any]]:
    """Load validation data from JSONL file."""
    data = []
    with open(validation_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def extract_prompts(data: List[Dict[str, Any]]) -> List[str]:
    """Extract prompts from validation data."""
    prompts = []
    for item in data:
        # Extract prompt content from the data structure
        if 'prompt' in item and isinstance(item['prompt'], list):
            # Join all messages into a single prompt
            prompt_parts = []
            for message in item['prompt']:
                if 'content' in message:
                    prompt_parts.append(message['content'])
            prompts.append('\n'.join(prompt_parts))
        elif 'prompt' in item and isinstance(item['prompt'], str):
            prompts.append(item['prompt'])
        elif 'problem' in item:
            prompts.append(item['problem'])
        else:
            raise ValueError(f"Unexpected prompt format in data: {item}")
    return prompts


def extract_ground_truths(data: List[Dict[str, Any]]) -> List[str]:
    """Extract ground truth answers from validation data."""
    ground_truths = []
    for item in data:
        if 'reward_model' in item and 'ground_truth' in item['reward_model']:
            ground_truths.append(item['reward_model']['ground_truth'])
        elif 'extra_info' in item and 'answer' in item['extra_info']:
            ground_truths.append(item['extra_info']['answer'])
        elif 'solution' in item:
            ground_truths.append(item['solution'])
        else:
            # If no ground truth found, use None
            ground_truths.append(None)
    return ground_truths


def evaluate_responses(
    responses: List[str],
    ground_truths: List[str],
    prompts: List[str] = None,
    extraction_method: str = "strict"
) -> Dict[str, Any]:
    """Evaluate responses against ground truths."""
    total = len(responses)
    correct = 0
    format_correct = 0
    no_answer = 0
    
    results = []
    failed_predictions = []
    
    for idx, (response, ground_truth) in enumerate(zip(responses, ground_truths)):
        # Extract solution from response
        extracted_answer = extract_solution(response, method=extraction_method)
        
        # Determine if answer is correct
        is_correct = False
        has_format = extracted_answer is not None
        
        if extracted_answer is None:
            no_answer += 1
            status = "no_answer"
        elif ground_truth is not None:
            gold = parse(ground_truth)
            answer = parse(extracted_answer)
            if verify(gold, answer):
                correct += 1
                format_correct += 1
                is_correct = True
                status = "correct"
            else:
                format_correct += 1
                status = "wrong_answer"
        else:
            status = "no_ground_truth"
        
        result = {
            "index": idx,
            "response": response,
            "extracted_answer": extracted_answer,
            "ground_truth": ground_truth,
            "is_correct": is_correct,
            "has_format": has_format,
            "status": status
        }
        
        if prompts is not None and idx < len(prompts):
            result["prompt"] = prompts[idx]
        
        results.append(result)
        
        # Save failed predictions
        if not is_correct and prompts is not None and idx < len(prompts):
            failed_predictions.append({
                "index": idx,
                "prompt": prompts[idx],
                "response": response,
                "extracted_answer": extracted_answer,
                "ground_truth": ground_truth,
                "status": status
            })
    
    # Calculate metrics
    accuracy = correct / total if total > 0 else 0.0
    format_rate = format_correct / total if total > 0 else 0.0
    no_answer_rate = no_answer / total if total > 0 else 0.0
    
    metrics = {
        "total_samples": total,
        "correct_answers": correct,
        "format_correct": format_correct,
        "no_answer": no_answer,
        "accuracy": accuracy,
        "format_rate": format_rate,
        "no_answer_rate": no_answer_rate
    }
    
    return {"metrics": metrics, "detailed_results": results, "failed_predictions": failed_predictions}


def find_model_config(model_path: str) -> str:
    """Find the directory containing config.json in model_path or its subdirectories.
    
    Args:
        model_path: Base path to search for config.json
        
    Returns:
        Path to directory containing config.json, or original model_path if not found
    """
    model_path = Path(model_path)
    
    # Check if config.json exists in the current directory
    if (model_path / "config.json").exists():
        print(f"Found config.json in: {model_path}")
        return str(model_path)
    
    # Search recursively in subdirectories
    print(f"Searching for config.json in subdirectories of: {model_path}")
    for config_file in model_path.rglob("config.json"):
        config_dir = config_file.parent
        print(f"Found config.json in: {config_dir}")
        return str(config_dir)
    
    # If not found, return original path with a warning
    print(f"Warning: config.json not found in {model_path} or its subdirectories")
    print(f"Using original model_path: {model_path}")
    return str(model_path)


def get_azureml_run():
    """Get AzureML Run context if available."""
    try:
        from azureml.core.run import Run
        azureml_run = Run.get_context()
        if azureml_run and "OfflineRun" not in azureml_run.id:
            return azureml_run
    except ImportError:
        print("Warning: azureml-core not available - AzureML logging disabled")
    except Exception as e:
        print(f"Warning: Failed to get AzureML run context: {e}")
    return None


def main():
    parser = argparse.ArgumentParser(description="vLLM Evaluation Component")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--validation_file", type=str, required=True, help="Path to validation JSONL file")
    parser.add_argument("--max_prompt_length", type=int, default=2048, help="Maximum prompt length")
    parser.add_argument("--max_response_length", type=int, default=1024, help="Maximum response length")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for evaluation")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="GPU memory utilization")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--extraction_method", type=str, default="strict", choices=["strict", "flexible"])
    parser.add_argument("--n_gpus_per_node", type=int, default=1, help="Number of GPUs per node")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for results")
    
    args = parser.parse_args()
    
    # Initialize AzureML Run context
    azureml_run = get_azureml_run()
    if azureml_run:
        print("AzureML Run context found - metrics will be logged to AzureML")
    
    # Find the actual model directory containing config.json
    args.model_path = find_model_config(args.model_path)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("vLLM Evaluation Component")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Validation file: {args.validation_file}")
    print(f"Extraction method: {args.extraction_method}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Load validation data
    print("\n[1/4] Loading validation data...")
    validation_data = load_validation_data(args.validation_file)
    print(f"Loaded {len(validation_data)} samples")
    
    # Extract prompts and ground truths
    print("\n[2/4] Extracting prompts and ground truths...")
    prompts = extract_prompts(validation_data)
    ground_truths = extract_ground_truths(validation_data)
    print(f"Extracted {len(prompts)} prompts")
    
    # Initialize vLLM
    print("\n[3/4] Initializing vLLM and generating responses...")
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=args.dtype,
        trust_remote_code=True
    )
    
    # Set sampling parameters
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_response_length
    )
    
    # Generate responses
    print(f"Generating responses with batch_size={args.batch_size}...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Extract generated texts
    responses = [output.outputs[0].text for output in outputs]
    print(f"Generated {len(responses)} responses")
    
    # Evaluate responses
    print("\n[4/4] Evaluating responses...")
    evaluation_results = evaluate_responses(responses, ground_truths, prompts, args.extraction_method)
    
    # Print metrics
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    metrics = evaluation_results["metrics"]
    print(f"Total Samples:        {metrics['total_samples']}")
    print(f"Correct Answers:      {metrics['correct_answers']}")
    print(f"Format Correct:       {metrics['format_correct']}")
    print(f"No Answer:            {metrics['no_answer']}")
    print(f"Accuracy:             {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Format Rate:          {metrics['format_rate']:.4f} ({metrics['format_rate']*100:.2f}%)")
    print(f"No Answer Rate:       {metrics['no_answer_rate']:.4f} ({metrics['no_answer_rate']*100:.2f}%)")
    print("=" * 80)
    
    # Log metrics to AzureML
    if azureml_run:
        print("\nLogging metrics to AzureML...")
        try:
            azureml_run.log("eval/accuracy", metrics['accuracy'])
            azureml_run.log("eval/format_rate", metrics['format_rate'])
            azureml_run.log("eval/no_answer_rate", metrics['no_answer_rate'])
            azureml_run.log("eval/total_samples", metrics['total_samples'])
            azureml_run.log("eval/correct_answers", metrics['correct_answers'])
            azureml_run.log("eval/format_correct", metrics['format_correct'])
            azureml_run.log("eval/no_answer", metrics['no_answer'])
            print("Successfully logged metrics to AzureML")
        except Exception as e:
            print(f"Warning: Failed to log metrics to AzureML: {e}")
    
    # Save results
    print("\nSaving results...")
    
    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save detailed results
    with open(os.path.join(args.output_dir, "detailed_results.jsonl"), 'w') as f:
        for result in evaluation_results["detailed_results"]:
            f.write(json.dumps(result) + '\n')
    
    # Save failed predictions
    failed_predictions = evaluation_results.get("failed_predictions", [])
    if failed_predictions:
        with open(os.path.join(args.output_dir, "failed_predictions.jsonl"), 'w') as f:
            for failed in failed_predictions:
                f.write(json.dumps(failed) + '\n')
        print(f"\nSaved {len(failed_predictions)} failed predictions to failed_predictions.jsonl")
    
    # Save summary
    summary = {
        "config": vars(args),
        "metrics": metrics
    }
    with open(os.path.join(args.output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {args.output_dir}")
    print("  - metrics.json: Evaluation metrics")
    print("  - detailed_results.jsonl: Detailed per-sample results")
    if failed_predictions:
        print("  - failed_predictions.jsonl: Failed predictions with prompts")
    print("  - summary.json: Complete summary with config and metrics")
    print("\nEvaluation completed successfully!")


if __name__ == "__main__":
    main()
